import torch
from models import LeNet, Discriminator
from preprocessing import get_loader
from trainer import train_source,evaluate,adapt_target_domain
from log import prepare_logger
from config import get_config


def main(config):
	logger = prepare_logger(config)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# get loaders
	if not config.is_train_source:
	    target_loader = get_loader(type="MNIST",train=False,batch_size=config.batch_size)

	source_train_loader = get_loader(type="SVHN",train=True,batch_size=config.batch_size)
	source_test_loader = get_loader(type="SVHN",train=False,batch_size=config.batch_size)

	
	# build source classifier
	model_src = LeNet(config.num_gpus).to(device)
	if (not config.is_train_source) or config.is_finetune:
		model_src.load_state_dict(torch.load(config.model_dir))

	# train source classifier
	if config.is_train_source:
		logger.info("train source classifier..")
		train_source(model_src,source_train_loader,source_test_loader,config,logger)
		logger.info("evaluate source classifier..")
		logger.info("test accurracy in source domain: %f\n" %(evaluate(model_src,source_test_loader)))

	else:
		# initialize target classifer with source classifer
		model_trg = torch.load(open("./pretrained/lenet-source.pth","rb"))

		# build discriminator
		D = Discriminator(config.num_gpus)

		# adaptation process
		logger.info("start adaptation process..")
		adapt_target_domain( D,model_src,model_trg,
		                    source_train_loader,target_loader, config )
		logger.info("evaluate target classifier..")
		logger.info("accurracy in target domain: %f\n" %(evaluate(model_trg,target_loader)))

if __name__ == '__main__':
	config, unparsed = get_config()
	main(config)
