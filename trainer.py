import torch
from torch import optim
from torch import nn
from torch.nn.init import xavier_normal_

import os

def weights_init(m):
    if isinstance(m,nn.Conv2d):
        xavier_normal_(m.weight.data)
        m.bias.data.fill_(0)
    elif isinstance(m,nn.Linear):
        xavier_normal_(m.weight.data)
        m.bias.data.fill_(0)
        

def _calculate_accuracy(logit,label):
    pred = torch.argmax(logit,dim=1)
    label = label.squeeze(dim=0)
    right = torch.sum(pred==label).item()
    acc = right / label.shape[0]
    return acc


def evaluate(model,loader):
    total_examples = 0
    right = 0
    
    for data,label in loader:
        _,logit = model(data)
        right += _calculate_accuracy(logit,label)*label.shape[0]
        total_examples += label.shape[0]
        
    return right/total_examples
        

def train_source(model,train_loader,test_loader,config,logger):
    """
    train classifier in source domiain
    
    model: source classifier model
    train_loader: loader containing training examples in source domain
    test_loader: loader containing test examples in source domain
    config: configuration variable
    logger: logger variable
    """

    base_message = (
                "Epoch: {epoch:<3d} "
                "Step: {step:<4d} "
                "Train Loss: {loss:<.6} "
                "Train Acc: {acc:<.4%} "
                "Val Acc: {val_acc:<.4%} "
                )
    device = torch.device("cuda" if config.num_gpus>0 and torch.cuda.is_available() else "cpu")

    
    # apply xavier initialization
    if not config.is_finetune:
        model.apply(weights_init)
        
    # set optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.lr,
                           betas=(config.beta1,config.beta2),
                           weight_decay=config.weight_decay)
    
    # set max_val_score for model checkpoint
    max_val_score = evaluate(model,test_loader)

    for epoch in range(config.max_epoch):
        for step,(data,label) in enumerate(train_loader):
            data,label = data.to(device),label.to(device)
            
            # train source classifier
            optimizer.zero_grad()
            _,logit = model(data)
            loss = nn.CrossEntropyLoss()(logit,label)
            loss.backward()
            optimizer.step()
            
            # print training histroy
            if (step+1)%100 == 0:        
                acc = _calculate_accuracy(logit,label)
                val_accuracy = evaluate(model,test_loader)
                msg = base_message.format(epoch=epoch+1,step=step+1,loss=loss,acc=acc,val_acc=val_accuracy)
                
                logger.info(msg)
                
                if val_accuracy > max_val_score:
                    if not os.path.exists("./pretrained"):
                        os.mkdir("./pretrained")
                    torch.save(model.state_dict(),open("./pretrained/lenet-source.pth","wb"))
                    max_val_score = val_accuracy
                
                
def adapt_target_domain(discriminator,model_s,model_t,loader_s,loader_t,config,logger):
    """
    adaptation process
     - train discriminator with source and target embedding
     - train target model( or target encoder) to fool discriminator
    
    model_s: source classifier model
    model_t: target classifier model
    loader_s: loader containing examples in source domain
    loader_t: loader containing examples in target domain
    config: configuration variable
    logger: logger variable
    """
    
    # apply xavier initialization to target classifier
    if not config.is_finetune:
        model.apply(weights_init)

    device = torch.device("cuda" if config.num_gpus>0 and torch.cuda.is_available() else "cpu")
    base_message = (
                "Epoch: {epoch:<3d} "
                "Step: {step:<4d} "
                "acc: {loss:<.6} "
                "D_loss: {acc:<.4%} "
                "G_loss: {val_acc:<.4%} "
                )
        
    # set optimizer    
    d_optimizer = optim.RMSprop(discriminator.parameters(),lr=config.lr,weight_decay=config.weight_decay)
    g_optimizer = optim.RMSprop(model_t.parameters(),lr=config.lr,weight_decay=config.weight_decay)
    
    # set GAN labels
    label_true = torch.zeros(100).type(torch.LongTensor).to(device)
    label_false = torch.ones(100).type(torch.LongTensor).to(device)
    
    
    
    # loss criterion 
    criterion = nn.CrossEntropyLoss()
    
    # max accuracy for model checkpoint
    max_acc = evaluate(model_t,loader_t)
        
    for epoch in range(config.max_epoch):
        for step, ((data_s,_),(data_t,_)) in enumerate(zip(loader_s,loader_t)):
            
            # train discriminator
            d_optimizer.zero_grad()
            
            embedding_s,_ = model_s(data_s)
            embedding_t,_ = model_t(data_t)
            
            pred_s = discriminator(embedding_s.detach())
            pred_t = discriminator(embedding_t.detach())
            
            d_loss_s = criterion(pred_s,label_true)
            d_loss_t = criterion(pred_t,label_false)
            
            d_loss = d_loss_s+d_loss_t
            d_loss.backward() 
            d_optimizer.step()
            
            # train target classifer (or target encoder)
            g_optimizer.zero_grad()
            
            pred_t = discriminator(embedding_t)
            g_loss = criterion(pred_t,label_true)

            g_loss.backward()
            g_optimizer.step()
            
            # make check point
            if (step+1)%100 == 0:
                acc = evaluate(model_t,loader_t)
                msg = base_message.format(epoch=epoch+1,step=step+1,acc=acc,D_loss=d_loss, G_loss = g_loss)
                logger.info(msg)
                
                if acc > max_acc:
                    max_acc = acc 
                    torch.save(discriminator.state_dict(),open("./pretrained/discriminator.pth","wb"))
                    torch.save(model_t.state_dict(),open("./pretrained/lenet-target.pth","wb"))
