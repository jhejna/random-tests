import numpy as np
import flax
import jax
from jax import numpy as jnp
import tensorflow_datasets as tfds
from flax import linen as nn
from flax.training import train_state
import optax
from functools import partial
import collections

NUM_LABELS = 10
TRAIN_DOMAINS = [
    # [3, 5],
    # [1, 7],
    # [4, 6],
    # [3, 8],
    # [0, 8]
    [3, 5],
    [4],
]
NUM_DOMAINS = len(TRAIN_DOMAINS)

TEST_DOMAIN = [3, 5]
BATCH_SIZE = 128
LR = 0.001

def prepare_dataset(split, domains):
    imgs, labels = tfds.as_numpy(tfds.load(
        'mnist',
        split=split,
        batch_size=-1,
        as_supervised=True,
    ))

    data_dict = {}
    for label in range(NUM_LABELS):
        data_dict[label] = imgs[labels == label]

    label_usages = collections.defaultdict(int)
    for domain in domains:
        for label in domain:
            label_usages[label] += 1
    
    # Assume no duplicates in domains.
    domain_size = min([v.shape[0] for v in data_dict.values()]) // NUM_DOMAINS
    xs, ys, domain_ids = [], [], []
    for i, domain in enumerate(domains):
        for label in domain:
            x = data_dict[label][i*domain_size:(i+1)*domain_size]
            y = np.full(x.shape[0], label, dtype=np.int64)
            domain_id = np.full(x.shape[0], i , dtype=np.int64)
            xs.append(x)
            ys.append(y)
            domain_ids.append(domain_id)

    xs = np.concatenate(xs, axis=0)
    ys = np.concatenate(ys, axis=0)
    domain_ids = np.concatenate(domain_ids, axis=0)
    assert xs.shape[0] == ys.shape[0] == domain_ids.shape[0]
    return xs, ys, domain_ids

def make_iterator(dataset, batch_size):
    x, y, domain_id = dataset
    dataset_size = x.shape[0]
    while True:
        idxs = np.random.permutation(dataset_size)
        for i in range(dataset_size // batch_size):
            batch_idxs = idxs[i*batch_size:(i+1)*batch_size]
            yield x[batch_idxs], y[batch_idxs], domain_id[batch_idxs]


class MLP(nn.Module):

    hidden_dims: tuple[int, ...] = (512,)
    output_dim: int = 10

    @nn.compact
    def __call__(self, x):
        x = jnp.reshape(x, (x.shape[0], -1))
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)
        return nn.Dense(self.output_dim)(x)

class TrainState(train_state.TrainState):
    alpha: jax.Array
    average_alpha: jax.Array

def _compute_per_domain_losses(losses, domains):
    one_hot_domains = jax.nn.one_hot(domains, NUM_DOMAINS, axis=0)  # (D, B)
    per_domain_losses = jnp.dot(one_hot_domains, losses)  # (D, B) dot (B,) -> D
    # count the number of losses for each domain
    norm = jnp.dot(one_hot_domains, losses != 0)
    norm = jnp.maximum(norm, 1.0)  # don't nan if there are no losses for a domain
    return per_domain_losses / norm

if __name__ == "__main__":

    train_ds = prepare_dataset("train", TRAIN_DOMAINS)
    val_ds = prepare_dataset("test[:50%]", [TEST_DOMAIN])
    test_ds = prepare_dataset("test[50%:]", [TEST_DOMAIN])
    
    rng = jax.random.PRNGKey(0)

    model = MLP()
    train_iter = make_iterator(train_ds, BATCH_SIZE)
    val_iter = make_iterator(val_ds, BATCH_SIZE)
    test_iter = make_iterator(test_ds, BATCH_SIZE)

    sample_x, _, _ = next(train_iter)

    rng, init_rng = jax.random.split(rng)
    params = model.init(init_rng, sample_x)

    tx = optax.adamw(5e-4, weight_decay=0.01)

    alpha = jnp.ones(NUM_DOMAINS, dtype=jnp.float32) / NUM_DOMAINS

    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx, alpha=alpha, average_alpha=alpha)

    # Now we can run the optimization
    @jax.jit
    def train_step(state, train_batch, test_batch):
        x_train, y_train, domain_id = train_batch
        x_test, y_test, _ = test_batch

        def per_example_loss(params, x, y):
            logits = state.apply_fn(params, x)
            return optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y)
        
        def per_domain_loss(params, x, y, domain_id):
            per_example_losses = per_example_loss(params, x, y)
            per_domain_loss = _compute_per_domain_losses(per_example_losses, domain_id)
            return per_domain_loss

        # Now we have a function that gives us the grad for every domain.
        per_domain_losses, per_domain_bwd = jax.vjp(partial(per_domain_loss, x=x_train, y=y_train, domain_id=domain_id), state.params)
        
        loss_grad_fn = jax.value_and_grad(lambda l, a: jnp.dot(a, l))

        # Get the per domain gradients. We can construct this one domain at a time.
        # this is essntially building the jacobian one row at a time.
        per_domain_grads = []
        for i in range(NUM_DOMAINS):
            one_hot = jax.nn.one_hot(jnp.full((), i, dtype=jnp.int32), NUM_DOMAINS)
            _, loss_grad = loss_grad_fn(per_domain_losses, one_hot)
            domain_grads = per_domain_bwd(loss_grad)[0]
            per_domain_grads.append(domain_grads)
        
        # Shapes are (grad..., D)
        per_domain_grads = jax.tree.map(lambda *args: jnp.stack(args, axis=-1), *per_domain_grads)

        # Next, we need to compute the per_domain gradient updates.
        def alpha_loss(alpha, params, per_domain_grads):
            # compute the new params as weighted gradients
            grads = jax.tree.map(lambda g: jnp.sum(g * alpha, axis=-1), per_domain_grads)
            new_params = jax.tree.map(lambda p, g: p - LR * g, params, grads)
            return per_example_loss(new_params, x_test, y_test).mean()

        # Update Alpha
        test_loss, alpha_grad = jax.value_and_grad(alpha_loss)(state.alpha, state.params, per_domain_grads)

        # Currently this is NaN.
        alpha = state.alpha * jnp.exp(-LR *  alpha_grad)
        alpha = alpha / jnp.sum(alpha)

        train_loss, loss_grad = loss_grad_fn(per_domain_losses, alpha)
        grads = per_domain_bwd(loss_grad)[0]
        
        average_alpha = state.average_alpha + (alpha - state.average_alpha) / (state.step + 1)
        state = state.apply_gradients(grads=grads, alpha=alpha, average_alpha=average_alpha)
        return state, train_loss, test_loss
    
    @jax.jit
    def val_step(state, batch):
        x, y, _ = batch
        logits = state.apply_fn(state.params, x)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y).mean()
        accuracy = (jnp.argmax(logits, axis=-1) == y).mean()
        return loss, accuracy

    train_losses = []
    val_losses = []
    average_alphas = []
    test_losses, test_accuracies = [], []
    for i in range(1000):
        train_batch = next(train_iter)
        val_batch = next(val_iter)
        test_batch = next(test_iter)
        state, train_loss, val_loss = train_step(state, train_batch, val_batch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        average_alphas.append(state.average_alpha)
        # Then compute the val loss
        test_loss, test_accuracy = val_step(state, test_batch)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        print("step", i)

    average_alphas = np.stack(average_alphas, axis=1)

    from matplotlib import pyplot as plt
    plt.plot(train_losses, label="train")
    plt.plot(test_losses, label="test")
    plt.plot(val_losses, label="val")
    plt.legend()
    plt.show()

    plt.plot(test_accuracies, label="acc")
    plt.show()

    for i in range(NUM_DOMAINS):
        plt.plot(average_alphas[i], label=str(TRAIN_DOMAINS[i]))
    plt.legend()
    plt.title("Test Domain: " + str(TEST_DOMAIN))
    plt.show()




    