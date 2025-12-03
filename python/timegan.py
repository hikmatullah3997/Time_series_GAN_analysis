# timegan.py - Updated for TensorFlow 2.x
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tqdm import tqdm

def timegan(ori_data, parameters):
    """TimeGAN function updated for TensorFlow 2.x compatibility."""
    
    # Initialize variables
    tf.compat.v1.disable_eager_execution()
    no, seq_len, dim = np.asarray(ori_data).shape
    
    # Normalize the data (already done in preprocessing, but just to be safe)
    def MinMaxScaler(data):
        """Min Max normalizer.
        
        Args:
        - data: original data
        
        Returns:
        - norm_data: normalized data
        """
        numerator = data - np.min(data, 0)
        denominator = np.max(data, 0) - np.min(data, 0)
        norm_data = numerator / (denominator + 1e-7)
        return norm_data
    
    ori_data = MinMaxScaler(ori_data)
    
    # Network Parameters
    hidden_dim   = parameters['hidden_dim']
    num_layers   = parameters['num_layer']
    iterations   = parameters['iterations']
    batch_size   = parameters['batch_size']
    module_name  = parameters['module']
    z_dim        = dim
    gamma        = 1
    
    # Create TF 1.x compatible graph
    tf.compat.v1.reset_default_graph()
    
    # Input placeholders
    X = tf.compat.v1.placeholder(tf.float32, [None, seq_len, dim], name = "myinput_x")
    Z = tf.compat.v1.placeholder(tf.float32, [None, seq_len, z_dim], name = "myinput_z")
    
    # Embedder function
    def embedder(X, reuse=False):
        with tf.compat.v1.variable_scope("embedder", reuse=reuse):
            e_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([
                tf.compat.v1.nn.rnn_cell.GRUCell(hidden_dim) for _ in range(num_layers)
            ])
            e_outputs, e_last_states = tf.compat.v1.nn.dynamic_rnn(
                e_cell, X, dtype=tf.float32, scope='embedder_rnn'
            )
            e_dense = tf.compat.v1.layers.dense(e_outputs, hidden_dim, activation=tf.nn.sigmoid)
        return e_dense
    
    # Recovery function  
    def recovery(H, reuse=False):
        with tf.compat.v1.variable_scope("recovery", reuse=reuse):
            r_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([
                tf.compat.v1.nn.rnn_cell.GRUCell(hidden_dim) for _ in range(num_layers)
            ])
            r_outputs, r_last_states = tf.compat.v1.nn.dynamic_rnn(
                r_cell, H, dtype=tf.float32, scope='recovery_rnn'
            )
            r_dense = tf.compat.v1.layers.dense(r_outputs, dim, activation=tf.nn.sigmoid)
        return r_dense
    
    # Generator function
    def generator(Z, reuse=False):
        with tf.compat.v1.variable_scope("generator", reuse=reuse):
            g_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([
                tf.compat.v1.nn.rnn_cell.GRUCell(hidden_dim) for _ in range(num_layers)
            ])
            g_outputs, g_last_states = tf.compat.v1.nn.dynamic_rnn(
                g_cell, Z, dtype=tf.float32, scope='generator_rnn'
            )
            g_dense = tf.compat.v1.layers.dense(g_outputs, hidden_dim, activation=tf.nn.sigmoid)
        return g_dense
    
    # Supervisor function
    def supervisor(H, reuse=False):
        with tf.compat.v1.variable_scope("supervisor", reuse=reuse):
            s_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([
                tf.compat.v1.nn.rnn_cell.GRUCell(hidden_dim) for _ in range(num_layers-1)
            ])
            s_outputs, s_last_states = tf.compat.v1.nn.dynamic_rnn(
                s_cell, H, dtype=tf.float32, scope='supervisor_rnn'
            )
            s_dense = tf.compat.v1.layers.dense(s_outputs, hidden_dim, activation=tf.nn.sigmoid)
        return s_dense
    
    # Discriminator function
    def discriminator(H, reuse=False):
        with tf.compat.v1.variable_scope("discriminator", reuse=reuse):
            d_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([
                tf.compat.v1.nn.rnn_cell.GRUCell(hidden_dim) for _ in range(num_layers)
            ])
            d_outputs, d_last_states = tf.compat.v1.nn.dynamic_rnn(
                d_cell, H, dtype=tf.float32, scope='discriminator_rnn'
            )
            d_dense = tf.compat.v1.layers.dense(d_outputs, 1, activation=None)
        return d_dense
    
    # Embedder & Recovery
    H = embedder(X)
    X_tilde = recovery(H)
    
    # Generator
    E_hat = generator(Z)
    H_hat = supervisor(E_hat)
    H_hat_superv = supervisor(H, reuse=True)
    
    # Synthetic data
    X_hat = recovery(H_hat, reuse=True)
    
    # Discriminator
    Y_fake = discriminator(H_hat)
    Y_real = discriminator(H, reuse=True)
    Y_fake_e = discriminator(E_hat, reuse=True)
    
    # Loss functions
    # Generator loss
    G_loss_U = tf.compat.v1.losses.sigmoid_cross_entropy(
        tf.ones_like(Y_fake), Y_fake
    )
    G_loss_U_e = tf.compat.v1.losses.sigmoid_cross_entropy(
        tf.ones_like(Y_fake_e), Y_fake_e
    )
    G_loss_S = tf.compat.v1.losses.mean_squared_error(H[:,1:,:], H_hat_superv[:,:-1,:])
    G_loss_V1 = tf.compat.v1.reduce_mean(
        tf.compat.v1.abs(tf.sqrt(tf.compat.v1.nn.moments(X_hat, [0])[1] + 1e-6) - 
                        tf.sqrt(tf.compat.v1.nn.moments(X, [0])[1] + 1e-6))
    )
    G_loss_V2 = tf.compat.v1.reduce_mean(
        tf.compat.v1.abs((tf.compat.v1.nn.moments(X_hat, [0])[0]) - 
                        (tf.compat.v1.nn.moments(X, [0])[0]))
    )
    G_loss_V = G_loss_V1 + G_loss_V2
    
    G_loss = G_loss_U + gamma * G_loss_U_e + 100 * tf.sqrt(G_loss_S) + 100 * G_loss_V
    
    # Discriminator loss
    D_loss_real = tf.compat.v1.losses.sigmoid_cross_entropy(tf.ones_like(Y_real), Y_real)
    D_loss_fake = tf.compat.v1.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake), Y_fake)
    D_loss_fake_e = tf.compat.v1.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake_e), Y_fake_e)
    D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e
    
    # Optimizers
    learning_rate = parameters['learning_rate']
    G_solver = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(
        G_loss, var_list=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='generator') + 
                        tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='supervisor')
    )
    D_solver = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(
        D_loss, var_list=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
    )
    E_solver = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(
        G_loss, var_list=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='embedder') + 
                        tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='recovery')
    )
    
    # Training
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    
    # Training loop
    for itt in tqdm(range(iterations)):
        # Train generator (twice)
        for _ in range(2):
            Z_mb = np.random.uniform(0, 1, [batch_size, seq_len, z_dim])
            _, step_g_loss = sess.run(
                [G_solver, G_loss], 
                feed_dict={Z: Z_mb, X: ori_data[np.random.randint(0, no, batch_size)]}
            )
        
        # Train embedder
        _, step_e_loss = sess.run(
            [E_solver, G_loss], 
            feed_dict={X: ori_data[np.random.randint(0, no, batch_size)]}
        )
        
        # Train discriminator
        Z_mb = np.random.uniform(0, 1, [batch_size, seq_len, z_dim])
        _, step_d_loss = sess.run(
            [D_solver, D_loss], 
            feed_dict={X: ori_data[np.random.randint(0, no, batch_size)], Z: Z_mb}
        )
    
    # Generate synthetic data
    synthetic_data = []
    for i in range(0, no, batch_size):
        Z_mb = np.random.uniform(0, 1, [min(batch_size, no-i), seq_len, z_dim])
        synthetic_batch = sess.run(X_hat, feed_dict={Z: Z_mb})
        synthetic_data.append(synthetic_batch)
    
    synthetic_data = np.vstack(synthetic_data)
    
    # Close session
    sess.close()
    
    return synthetic_data