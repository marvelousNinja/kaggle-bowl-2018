from tqdm import tqdm

def fit_model(model, train_generator, validation_generator, optimizer, loss_fn, num_epochs, num_batches, after_validation):
    for _ in tqdm(range(num_epochs)):
        for _ in tqdm(range(num_batches)):
            optimizer.zero_grad()
            inputs, gt = next(train_generator)
            loss = loss_fn(model(inputs), gt)
            loss.backward()
            optimizer.step()

        inputs, gt = next(validation_generator)
        outputs = model.eval()(inputs)
        loss = loss_fn(outputs, gt)
        tqdm.write(f'val loss {loss.data[0]:.5f}')
        if after_validation: after_validation(inputs, outputs, gt)
