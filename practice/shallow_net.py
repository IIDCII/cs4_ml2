import numpy as np

# 1 with 2 features (assuming features is columns)
inputs = np.array([[0.05, 0.1]])
y = np.array([0.67, 0.33])
test_data = np.array([[667, 43]])

# bias set to col 0
weights = np.array(
    [[[0.35, 0.15, 0.25],
     [0.35, 0.2, 0.3]],

     [[0.6, 0.4, 0.5],
      [0.6, 0.45, 0.55]]]
    )

def sigmoid(x):
    return 1/(1+np.exp(-x))

def mse(y, y_hat):
    return np.mean((y - y_hat)**2)

# manual forward pass
def forward_pass(inputs, weights):
    f0 = (weights[0,:,1:] @ inputs[0].T) + weights[0,:,0]
    h = sigmoid(f0)
    f1 = (weights[1,:,1:] @ h) + weights[1,:,0]
    y_hat = sigmoid(f1)
    loss = mse(y, y_hat)

    return y_hat, loss, h, f0, f1

# manual backward pass
def backward_pass(inputs, weights, y, y_hat, h, f0, f1):
    dL_dy_hat = y_hat - y
    dy_hat_df1 = sigmoid(f1) * (1 - sigmoid(f1))
    df1_dh = weights[1,:,1:]
    dh_df0 = sigmoid(f0) * (1 - sigmoid(f0))

    dL_dw2 = np.outer(dy_hat_df1 * dL_dy_hat, h)
    dL_dw1 = np.outer(dh_df0 * (dy_hat_df1 * dL_dy_hat) @ weights[1,:,1:] * dL_dy_hat, inputs[0])

    dL_db2 = dy_hat_df1 * dL_dy_hat
    dL_db1 = dh_df0 * (dy_hat_df1 * dL_dy_hat) @ weights[1,:,1:] * dL_dy_hat

    # add bis to 1st col in dL_dw
    dL_dw1 = np.column_stack([dL_db1, dL_dw1])
    dL_dw2 = np.column_stack([dL_db2, dL_dw2])

    dL_dw = np.array([dL_dw1, dL_dw2])

    return dL_dw

# update weights
def update_weights(weights, dL_dw, lr = 0.5):
    return weights - lr * dL_dw

# training
def training(inputs, weights, y, lr = 0.5, epochs = 1000):
    for epoch in range(epochs):
        lr = lr * 0.999
        y_hat, loss, h, f0, f1 = forward_pass(inputs, weights)
        dL_dw = backward_pass(inputs, weights, y, y_hat, h, f0, f1)
        weights = update_weights(weights, dL_dw, lr)
        if epoch % 100 == 0:
            print ("epoch: ", epoch, "loss: ", loss)
    return weights

def testing(inputs, weights):
    y_hat, loss, h, f0, f1 = forward_pass(inputs, weights)
    print("y_hat: ", y_hat)
    print("loss: ", loss)

weights = training(inputs, weights, y)
testing(test_data, weights)