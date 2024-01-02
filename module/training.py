import tensorflow as tf

class TensorflowTrainer:

    def __init__(
        self,
        model:tf.keras.models.Model,
        optimizer:tf.keras.optimizers=None,
        loss:tf.keras.losses=None,
        train_acc_metric:tf.keras.metrics=None,
        test_acc_metric:tf.keras.metrics=None,
        epochs:int=1,
        batch_size:int=32
    ) -> None:
        # Define model with its hyper-parameter
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size

        # Accuracy metric
        self.train_accuracy_metric = train_acc_metric
        self.test_accuracy_metric = test_acc_metric

        # Define empty train and test dataset
        self.train_dataset, self.test_dataset = None, None

        # Define empty gradient information
        self.gradient_records = []
        self.early_loss = []

        # Define accuracy and loss list
        self.train_accuracy_list, self.train_loss_list = [], []
        self.test_accuracy_list, self.test_loss_list = [], []

    def prepare_dataset(self, dataset_tuple:tuple, num_classes=3, categorical=False) -> None:
        # Unpack the data from tuple
        x_train, y_train, x_test, y_test = dataset_tuple

        if categorical:
            # Convert the label into categorical
            y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
            y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

        # Prepare training dataset
        self.train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        self.train_dataset = self.train_dataset.shuffle(buffer_size=1024).batch(self.batch_size)
        self.test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        self.test_dataset = self.test_dataset.batch(self.batch_size)

    @tf.function
    def _train_step(self, x, y):
        with tf.GradientTape() as tape:
            # Train the model and get the logits
            logits = self.model(x, training=True)
            # Calculate the loss using weight/logits information
            loss = self.loss(y, logits)

        # Use the gradient tape to retrieve trainable variable
        gradients = tape.gradient(loss, self.model.trainable_weights)
        # Update the gradient with the help of defined optimizer
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))

        # Log and Update the training metrics
        self.train_accuracy_metric.update_state(y, logits)

        # Return the loss calculation
        return loss

    @tf.function
    def _train_step_with_gradient(self, x, y):
        with tf.GradientTape() as tape:
            # Train the model and get the logits
            logits = self.model(x, training=True)
            # Calculate the loss using weight/logits information
            loss = self.loss(y, logits)

        # Use the gradient tape to retrieve trainable variable
        gradients = tape.gradient(loss, self.model.trainable_weights)
        # Update the gradient with the help of defined optimizer
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))

        # Log and Update the training metrics
        self.train_accuracy_metric.update_state(y, logits)

        # Return the loss calculation
        return loss, gradients

    @tf.function
    def _test_step(self, x, y):
        # Only get the logits for validation purpose
        # Inference logits information
        logits = self.model(x, training=False)

        # Get the loss from the logits
        loss = self.loss(y, logits)

        # Update accuracy stat
        self.test_accuracy_metric.update_state(y, logits)

        return loss

    def record_weight(self, gradients, trainable_weights, loss):
        gradient_map = {}

        for gradient, weight in zip(gradients, trainable_weights):
            if '/kernel' not in weight.name:
                continue # Skip bias parameter
            
            weight_name = weight.name.split("/")[0]
            gradient_map[weight_name] = gradient.numpy()

            self.gradient_records.append(gradient_map)
            self.early_loss.append(loss.numpy())

    def train(self, logging_each_step:int=16) -> dict:
        # Set the device with GPU
        with tf.device('/gpu:0'):
            # Begin training, first loop is epochs
            for epoch in range(self.epochs):
                print("Start of the Epoch {}".format(epoch))
                # Define total of loss var for each epoch
                total_training_loss, total_testing_loss = 0, 0
                # Second Loop for batches
                for step, (x_batch_train, y_batch_train) in enumerate(self.train_dataset):
                    # If step is zero, call train step that return gradient informations
                    # This gradient information will be recorded
                    if step == 0:
                        train_step_loss, gradients = self._train_step_with_gradient(x_batch_train, y_batch_train)
                        # Record the gradient information
                        self.record_weight(gradients, self.model.trainable_weights, train_step_loss)
                    else:
                        train_step_loss = self._train_step(x_batch_train, y_batch_train)

                    # Accumulate training batch of loss
                    total_training_loss += train_step_loss.numpy()

                    # Verbose each training step
                    if step % logging_each_step == 0:
                        print("Training loss at step {} : {}".format(step, float(train_step_loss)))

                # Get the training result accuracy in one epochs
                train_acc = self.train_accuracy_metric.result()

                # Calculate the average of the loss
                average_training_loss = total_training_loss / self.batch_size

                # Perform batched evalation
                for x_batch_test, y_batch_test in self.test_dataset:
                    test_step_loss = self._test_step(x_batch_test, y_batch_test)
                    # Accumulate evaluation batch of loss
                    total_testing_loss += test_step_loss.numpy()

                # Evaluation average loss
                average_testing_loss = total_testing_loss / self.batch_size

                # Get test accuracy metrics
                test_acc = self.test_accuracy_metric.result()
                # Reset the accuracy metric state
                self.test_accuracy_metric.reset_state()

                # Verbose each epoch
                print(
                    "Epoch : {}. Training Accuracy : {} and Testing Accuracy : {}, Training Loss : {}, Testing Loss : {}"
                    .format(
                        epoch, 
                        float(train_acc), 
                        float(test_acc),
                        average_training_loss,
                        average_testing_loss
                    )
                )

                # Documented the accuracy and loss for each train and test set
                self.train_accuracy_list.append(float(train_acc))
                self.test_accuracy_list.append(float(test_acc))
                self.train_loss_list.append(average_training_loss)
                self.test_loss_list.append(average_testing_loss)

        return {
            'final_train_accuracy':self.train_accuracy_list[-1],
            'final_test_accuracy':self.test_accuracy_list[-1],
            'train_accuracy':self.train_accuracy_list,
            'test_accuracy':self.test_accuracy_list,
            'train_loss':self.train_loss_list,
            'test_loss':self.test_loss_list,
            'gradient_trace':self.gradient_records,
            'gradient_loss_trace':self.early_loss
        }