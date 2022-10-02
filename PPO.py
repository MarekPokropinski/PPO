import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, Input, Lambda
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop
import tensorflow_probability as tfp
from tensorflow_probability import distributions
import numpy as np
import random
import os
from gym.spaces import Discrete, Box
from gym.spaces.multi_discrete import MultiDiscrete
from collections import defaultdict

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# class Pd:
#     def __init__(self, )


class PPO:
    def __init__(self, observation_space, action_space, entropy_coeff=0.0):
        self.gamma = 0.99
        self.λ = 0.95
        self.model:Model = None
        self.action_space = action_space
        if len(observation_space.shape) == 2:
            model = build_mlp_model(
                observation_space.shape[1:])
        
        elif len(observation_space.shape) == 4:
            model = build_conv_model(
                observation_space.shape[1:])
        else:
            raise Exception(
                f'Unsupported observation space shape {observation_space.shape}')

        model_input_shape = model.input_shape
        input_tensor = Input(shape=model_input_shape[1:])
        value, latent = model(input_tensor)

        if issubclass(type(action_space), MultiDiscrete):
            pi = Dense(action_space.nvec[0])(latent)
            self.model = Model(input_tensor, [value, pi])
            self.get_pd = self.get_categorical_pd
        
        elif issubclass(type(action_space), Box):
            size = action_space.shape[1]
            # mean = Dense(size)(latent)
            # std = Dense(size, activation='exponential')(latent)
            # self.model = Model(input_tensor, [value, mean, std])
            b0 = Dense(size, activation='softplus')(latent)
            b1 = Dense(size, activation='softplus')(latent)
            b0 = Lambda(lambda x: 1+x)(b0)
            b1 = Lambda(lambda x: 1+x)(b1)
            self.model = Model(input_tensor, [value, b0, b1])
            self.get_pd = self.get_normal_pd
        
        else:
            raise Exception(
                f'Unsupported action space {type(action_space)}')

        self._train_steps = 0
        self.optim = tf.keras.optimizers.Adam(lr=0.0005)
        self.entropy_coeff = entropy_coeff
        self.model.summary()

    @tf.function
    def act(self, state):
        pd, value = self.get_pd(state)
        action = pd.sample()
        logp = pd.log_prob(action)
        return action, value, logp

    def value(self, state):
        pd, value = self.get_pd(state)
        return value

    def get_categorical_pd(self, obs):
        value, pi = self.model(obs)
        pd = distributions.Categorical(logits=pi)
        return pd, tf.squeeze(value, axis=-1)

    def get_normal_pd(self, obs):
        # value, mean, std = self.model(obs)
        # pd = distributions.MultivariateNormalDiag(loc=mean, scale_diag=std)
        value, b0, b1 = self.model(obs)
        pd = distributions.Beta(b0, b1)
        pd = distributions.Independent(pd, reinterpreted_batch_ndims=1)
        return pd, tf.squeeze(value, axis=-1)

    def get_pd(self, obs):
        raise NotImplementedError()
    
    def probas(self, obs):
        value, pi = self.model(obs)
        return tf.math.softmax(pi)

    @tf.function
    def get_loss_pi(self, adv, logp, old_logp, clip):
        ratio = tf.math.exp(logp-old_logp)
        clipped_adv = tf.clip_by_value(ratio, 1-clip, 1+clip)*adv
        loss_pi = -tf.reduce_mean(tf.minimum(ratio*adv, clipped_adv))
        return loss_pi

    @tf.function
    def get_loss_value(self, pred_value, returns):
        return tf.reduce_mean((pred_value-returns)**2)

    # @tf.function
    def grad(self, obs, clip, returns, values, actions, old_logp):
        epsilon = 1e-8
        adv = returns - values
        adv = (adv-tf.reduce_mean(adv)/(tf.keras.backend.std(adv)+epsilon))

        with tf.GradientTape() as tape:
            pd, pred_value = self.get_pd(obs)
            logp = pd.log_prob(actions)
            loss_pi = self.get_loss_pi(adv, logp, old_logp, clip)
            loss_v = self.get_loss_value(pred_value, returns)
            entropy = pd.entropy()

            loss = loss_pi+loss_v*0.5-entropy*self.entropy_coeff

        approxkl = .5 * tf.reduce_mean(tf.square(old_logp-logp))

        vars = tape.watched_variables()
        grads = tape.gradient(loss, vars)
        grads_and_vars = zip(grads, vars)
        # self.optim.apply_gradients(zip(grads, vars))
        return grads_and_vars, loss_pi, loss_v, entropy, approxkl

    @tf.function
    def train_on_batch(self, lr, cliprange, obs, returns, actions, values, old_logp):
        obs = tf.keras.backend.cast_to_floatx(obs)
        grads_and_vars, loss_pi, loss_v, entropy, approxkl = self.grad(
            obs, cliprange, returns, values, actions, old_logp)
        self.optim.learning_rate = lr
        self.optim.apply_gradients(grads_and_vars)

        return loss_pi, loss_v, entropy, approxkl

    def calculate_returns(self, rewards, values, dones, last_values, last_dones, dtype):
        last_adv = 0
        steps = len(rewards)
        nenvs = rewards[0].shape[0]
        adv = np.zeros((steps, nenvs))
        for t in reversed(range(steps)):
            if t == steps-1:
                next_non_terminal = 1.0 - last_dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - dones[t+1]
                next_values = values[t+1]

            delta = rewards[t] + self.gamma * \
                next_values * next_non_terminal - values[t]
            last_adv = delta + self.gamma * self.λ * next_non_terminal * last_adv
            adv[t] = last_adv
        return (adv + values).astype(dtype)

    def train(self, obs, returns, actions, values, old_logp, cliprange, lr, mb_size=32, epochs=1):
        # losses = {'policy_loss':}
        losses = defaultdict(list)
        nbatch = obs.shape[0]
        inds = np.arange(nbatch)
        for epoch in range(epochs):
            np.random.shuffle(inds)
            for start in range(0, nbatch, mb_size):
                end = start + mb_size
                mb_idx = inds[start:end]
                slices = (tf.constant(arr[mb_idx]) for arr in (
                    obs, returns, actions, values, old_logp))
                loss = self.train_on_batch(
                    lr, cliprange, *slices)
                for k, v in zip(['policy_loss', 'value_loss', 'entropy', 'kl'], loss):
                    losses[k].append(v)

        return losses

    def learn(self, env, nepisodes, steps_per_ep=2048, epochs_per_ep=4, mb_size=256, clip_range = 0.2, lr = 3e-4, save_interval=10, model_name='model', start_from_ep=0):
        dtype = tf.keras.backend.floatx()
        obs, _ = env.reset()
        dones = np.zeros((env.num_envs,), dtype=dtype)
        scores = np.zeros_like(dones)
        score_history = []
        if type(lr)==float:
            lr = lambda x: lr

        for e in range(start_from_ep, nepisodes):
            mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_logp = [], [], [], [], [] ,[]
            for _ in range(steps_per_ep):
                actions, values, logp = self.act(obs)
                actions = actions.numpy()
                values = values.numpy()
                logp = logp.numpy()

                mb_obs.append(obs)
                mb_actions.append(actions)
                mb_values.append(values)
                mb_dones.append(dones)
                mb_logp.append(logp)

                obs, rewards, terminated, truncated, _ = env.step(actions)
                dones = terminated
                dones = tf.cast(dones, tf.float32)

                mb_rewards.append(rewards)
                scores += rewards
                for i in range(rewards.shape[0]):
                    if dones[i]:
                        score_history.append(scores[i])
                        scores[i] = 0

            last_values = self.value(obs).numpy()

            returns = self.calculate_returns(mb_rewards, mb_values, mb_dones, last_values, dones, dtype=dtype)

            mb_obs = np.concatenate(mb_obs, axis=0)
            returns = np.concatenate(returns, axis=0)
            mb_actions = np.concatenate(mb_actions, axis=0)
            mb_values = np.concatenate(mb_values, axis=0)
            mb_logp = np.concatenate(mb_logp, axis=0)
            
            lr_now = lr(1.0 - e/nepisodes)
            # self.train(mb_obs, returns, mb_actions, mb_values, mb_logp, clip_range, lr_now, mb_size, epochs=epochs_per_ep)
            avg_score = np.mean(score_history[-300:] if len(score_history)>300 else score_history)
            if e%1==0:
                print(f'step {e}/{nepisodes}, avg score: {avg_score:.3f}')

            if e%save_interval:
                self.save(model_name)

    def save(self, filename='model'):
        self.model.save_weights(filename)
    
    def load(self, filename='model'):
        self.model.load_weights(filename)






def build_mlp_model(state_size) -> Model:
    input = Input(shape=state_size)

    h1 = Dense(256, activation='relu')(input)
    # model.add(Dropout(0.5))
    latent = Dense(256, activation='relu')(h1)
    # model.add(Dropout(0.5))
    value = Dense(1, activation='linear')(latent)
    # pi = Dense(actions_size, activation='linear')(h2)

    return Model(input, [value, latent])




def build_conv_model(state_size) -> Model:
    input = Input(shape=state_size)
    h1 = Conv2D(32, 8, 4, activation='relu')(input)
    h2 = Conv2D(64, 4, 2, activation='relu')(h1)
    h3 = Conv2D(64, 4, 2, activation='relu')(h2)
    flat = Flatten()(h3)
    latent = Dense(512, activation='relu')(flat)
    # pi = Dense(actions_size, activation='linear')(latent)
    value = Dense(1, activation='linear')(latent)
    
    return Model(input, [value, latent])

