import data_loader
import train
import validation

# load data
trainloader = data_loader.get_train_data_loader()
testloader = data_loader.get_test_data_loader()

print('Starting training....')

# train
train.trainYeroudj(trainloader, testloader)

print('Finished Training.')
print('Accuracy of the best model: ')

# final accuracy of the best model
validation.valTrainingSaveBest(0, testloader)

print("Finished.")

