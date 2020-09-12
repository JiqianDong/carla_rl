import torch
import torch.nn as nn
import json
import pickle

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"

def train_model_step(model,optimizer,loss_fn, x_batch, y_batch):
    model.train()
    training_loss = 0
    # forward pass
    outputs = model.forward(x_batch)
    loss = loss_fn(outputs,y_batch)
    loss_out = loss.item()
    # back propagate
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss_out

def train_one_epoch(cav_model, hdv_model, cav_optimizer, hdv_optimizer, dataset):

    cav_model.to(DEVICE)
    cav_loss_fn = nn.MSELoss().to(DEVICE)

    hdv_model.to(DEVICE)
    hdv_loss_fn = nn.MSELoss().to(DEVICE)


    batch_id = 0

    
    for state_batch, action_batch, next_state_batch in dataset.random_iterator():
        hdv_X = []
        hdv_Y = []
        for key in state_batch:
            if key!='CAV'and key!='current_control':
                hdv_X.append(state_batch[key])
                hdv_Y.append(next_state_batch[key])
        hdv_X = torch.stack(hdv_X, dim=0) #(batch_size*num_vehicles, window_size, feature_size)
        hdv_Y = torch.stack(hdv_Y, dim=0) 

        cav_X = state_batch['CAV'] # FIXME THERE IS A NEED OF FUSION ACTION WITH STATE
        cav_Y = next_state_batch['CAV']

        cav_loss = train_model_step(cav_model,cav_optimizer,cav_loss_fn, cav_X, cav_Y)
        hdv_loss = train_model_step(hdv_model,hdv_optimizer,hdv_loss_fn, hdv_X, hdv_Y)

        cav_training_loss += cav_loss
        hdv_training_loss += hdv_loss

    
        # print statistics
        if batch_id%50 == 0:
            print("Batch number {}: ------  cav loss: {} ---- hdv loss: {}".format(batch_id+1,cav_loss, hdv_loss))
        batch_id += 1

    cav_training_loss /= batch_id
    hdv_training_loss /= batch_id
    # # after each epoch, run through the validation dataset (if any)
    # model.eval()
    # valid_loss = None
    # if valid_data_loader:
    #     batch_id = 0
    #     valid_loss = 0
    #     for x_batch, y_batch in valid_data_loader:
    #         batch_id += 1
    #         outputs = model.forward(x_batch)
    #         loss = loss_fn(outputs,y_batch)
    #         valid_loss += loss.item()
    #     valid_loss /= batch_id # devide by each batch
    return cav_training_loss, hdv_training_loss#, valid_loss


def training_main(model_type, training=True):
    
    from traj_pred_models import RNN_Predictor, MLP_Predictor, LinearRegression

    cav_model_file = './models/{}_{}.pt'.format('cav', model_type)
    hdv_model_file = './models/{}_{}.pt'.format('hdv', model_type)
    ### MODEL PARAMETERS
    LEARNING_RATE = 0.01
    INPUT_DIM = 6
    OUTPUT_DIM = 6
    ENCODE_DIM = 16
    WINDOW_SIZE = 5
    BATCH_SIZE = 15
    NUM_TRAINING_EPOCHS = 1000

    # load the saved dataset
    with open('./experience_data/data_pickle.pickle','rb') as f:
        init_dataset = pickle.load(f) 
    

    # setup the model
    if model_type == 'rnn':
        cav_predictor = RNN_Predictor(input_dim=INPUT_DIM, encode_dim=ENCODE_DIM, output_dim=OUTPUT_DIM, return_sequence=False)
        # hdv_predictor = 
    elif model_type == 'mlp':
        HIDDEN_DIM = 32
        NUM_HIDDEN_LAYERS = 1
        cav_predictor = MLP_Predictor(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,output_dim=OUTPUT_DIM,num_hidden_layers=NUM_HIDDEN_LAYERS)
    elif model_type == 'linreg':
        cav_predictor = LinearRegression(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM)
    else:
        raise NotImplementedError("unknown model class, only rnn, mlp, linreg are supported")


    if training:
        optimizer = torch.optim.Adam(predictor.parameters() ,lr=LEARNING_RATE)
        losses = []
        print("start training %s ... \n"%model)
        for i in range(1,NUM_TRAINING_EPOCHS+1):
            if i%100 ==0: print(" ------ start epoch %d ------"%(1+i))
            training_loss = train_one_epoch(predictor, optimizer, init_dataset)
            losses.append(training_loss)
        
        # save the model
        torch.save(predictor.state_dict(),model_file)

        # save the training loss history
        logdir = './training_stats/'
        with open(logdir + model +'_training_loss.txt','w') as f:
            json.dump({"training_loss":losses}, f)
        
        # plot the training curve
        from generate_training_plots import loss_plot
        loss_plot(logdir)
    else:
        # load the model model.
        predictor.load_state_dict(torch.load(model_file))
        predictor.to(DEVICE) # make it on the device
        predictor.eval()



if __name__ == "__main__":
    models = ['mlp','rnn','linreg']
    # models = ['linreg']

    for model in models:
        training_main(model,"cav",training=True)
    



