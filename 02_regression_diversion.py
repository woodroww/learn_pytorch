# https://youtu.be/Z_ikDlimN6A?t=40979
# Begin a diversion back into regression
# ------------------------------------------------------------------------------
# So we know the problem is we don't have any non-linearity
# Daniel wants to trouble shoot this problem step by step
# One way to troubleshoot a larger problem is to test out a smaller problem
# Can the new model fit a linear regression? We want to see if the model
# is learning anything, but maybe not the circles solution

weight = 0.7
bias = 0.3
start = 0
end = 1
step = 0.01
X_regression = torch.arange(start, end, step).unsqueeze(dim=1)
y_regression = bias + weight * X_regression

(
    X_regression_train,
    X_regression_test,
    y_regression_train,
    y_regression_test,
) = train_test_split(
    X_regression, y_regression, test_size=0.20, shuffle=False, random_state=42
)

X_regression_train = X_regression_train.to(device)
y_regression_train = y_regression_train.to(device)
X_regression_test = X_regression_test.to(device)
y_regression_test = y_regression_test.to(device)
len(X_regression_train)
len(y_regression_train)
len(X_regression_test)
len(y_regression_test)

plot_predictions(
    train_data=X_regression_train,
    train_labels=y_regression_train,
    test_data=X_regression_test,
    test_labels=y_regression_test,
)


class RegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=1, out_features=20)
        self.layer_2 = nn.Linear(in_features=20, out_features=1)

    def forward(self, x):
        z = self.layer_1(x)
        z = self.layer_2(z)
        return z


# train model
regression_model = RegressionModel().to(device)
loss_fn = nn.L1Loss()
opt = torch.optim.SGD(regression_model.parameters(), lr=0.01)
for epoch in range(1000):
    regression_model.train()  # sets up the parameters the require gradients
    y_pred = regression_model(X_regression_train)
    loss = loss_fn(y_pred, y_regression_train)
    opt.zero_grad()
    loss.backward()
    opt.step()
    regression_model.eval()
    with torch.inference_mode():
        test_pred = regression_model(X_regression_test)
        test_loss = loss_fn(test_pred, y_regression_test)
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f} | Test loss: {test_loss}")

regression_model.eval()
with torch.inference_mode():
    y_preds = regression_model(X_regression_test)

plot_predictions(
    train_data=X_regression_train.cpu(),
    train_labels=y_regression_train.cpu(),
    test_data=X_regression_test.cpu(),
    test_labels=y_regression_test.cpu(),
    predictions=y_preds.cpu()
)

# https://youtu.be/Z_ikDlimN6A?t=42096
# end diversion
# ------------------------------------------------------------------------------

