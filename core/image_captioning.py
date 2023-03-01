import tensorflow as tf
from transformers import RobertaConfig, TFRobertaModel, PhobertTokenizer
from transformers import ViTFeatureExtractor, TFViTModel
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense

PHOBERT_NAME = 'vinai/phobert-base'

# See all ViT models at https://huggingface.co/models?filter=vit
VIT_MODELS = ["google/vit-base-patch32-384",
              "google/vit-base-patch32-224-in21k",
              "google/vit-base-patch16-224-in21k",
              "google/vit-base-patch16-224",
              "google/vit-base-patch16-384"]

# See all roberta models at https://huggingface.co/models?filter=roberta
ROBERTA_MODELS = ["vinai/phobert-base",
                  "vinai/phobert-large"]

TOKERNIZER = PhobertTokenizer.from_pretrained(PHOBERT_NAME)
CONFIG = {
    "pretrained_model": {
        "vit": VIT_MODELS[0],
        "roberta": ROBERTA_MODELS[0]
    },
    "tokenizer": TOKERNIZER
}



def GetRobertaDecoder(vocab_size, pretrained=PHOBERT_NAME): 
    configuration = RobertaConfig(is_decoder = True,
                                  add_cross_attention = True)
    model = TFRobertaModel(configuration)
    model.from_pretrained(pretrained)
    model.layers[0].submodules[1].trainable = False
    return model
    
def GetVitEncoder(pretrained_model):
    model = TFViTModel.from_pretrained(pretrained_model)
    model.layers[0].submodules[3].trainable = False
    return model

def GetViTPreprocess(pretrained_model):
    model = ViTFeatureExtractor.from_pretrained(pretrained_model)
    return model


def getCaptionModel():
    checkpoint_path = "model\base-384"
    caption_model = TransformerCaptioner(CONFIG)
    ckpt = tf.train.Checkpoint()
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print(f'Loaded checkpoint from {checkpoint_path}')
    return caption_model

class TransformerCaptioner(Model):
    
    def __init__(self, config):
        super().__init__()

        self.image_preprocessor = GetViTPreprocess(config["pretrained_model"]["vit"])
        self.image_encoder = GetVitEncoder(config["pretrained_model"]["vit"])

        self.tokenizer = config["tokenizer"]
        self.decoder = GetRobertaDecoder(self.tokenizer.vocab_size, config["pretrained_model"]["roberta"])

        self.token_classifier = Dense(units=self.tokenizer.vocab_size)
    
    def call(self, image, text, training=False):        
        encoder_hidden_states = self.image_encoder(**image, training=training).last_hidden_state
        decoder_output = self.decoder(encoder_hidden_states=encoder_hidden_states, **text, training=training)
        output = self.token_classifier(decoder_output.last_hidden_state)
        return output    
    
    
