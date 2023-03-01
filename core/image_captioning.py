import tensorflow as tf
from transformers import RobertaConfig, TFRobertaModel, PhobertTokenizer
from transformers import ViTFeatureExtractor, TFViTModel
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.losses import categorical_crossentropy
import os.path as osp
from core.train import train

def GetRobertaDecoder(pretrained): 
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

def load_weight():
    CKPT = tf.train.Checkpoint(model=MODEL)
    CKPT_MANAGER = tf.train.CheckpointManager(CKPT, CHECKPOINT_PATH, max_to_keep=5)
    CKPT.restore(CKPT_MANAGER.latest_checkpoint)
    print(f'Loaded checkpoint from {CHECKPOINT_PATH}')

class TransformerCaptioner(Model):
    
    def __init__(self, config):
        super().__init__()

        self.image_preprocessor = GetViTPreprocess(config["pretrained_model"]["vit"])
        self.image_encoder = GetVitEncoder(config["pretrained_model"]["vit"])

        self.tokenizer = config["tokenizer"]
        self.decoder = GetRobertaDecoder(config["pretrained_model"]["roberta"])

        self.token_classifier = Dense(units=self.tokenizer.vocab_size)
    
    def call(self, image, text, training=False):        
        encoder_hidden_states = self.image_encoder(**image, training=training).last_hidden_state
        decoder_output = self.decoder(encoder_hidden_states=encoder_hidden_states, **text, training=training)
        output = self.token_classifier(decoder_output.last_hidden_state)
        return output    
    

PHOBERT_NAME = 'vinai/phobert-base'
CHECKPOINT_PATH =osp.join("model", "base-384")


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

def scce_with_ls(y, y_hat):
    y = tf.one_hot(tf.cast(y, tf.int32), TOKERNIZER.vocab_size)
    return categorical_crossentropy(y, y_hat, from_logits=True)

loss_object = scce_with_ls

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 1))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask
  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

LEARNING_RATE = 2e-5
LOSS = loss_function
OPTIMIZER = tf.keras.optimizers.legacy.Adam(LEARNING_RATE)
MODEL = TransformerCaptioner(CONFIG)
MODEL.compile(optimizer='adam', loss=LOSS)
CKPT = tf.train.Checkpoint(model=MODEL,
                           optimizer=OPTIMIZER)

CKPT_MANAGER = tf.train.CheckpointManager(CKPT, CHECKPOINT_PATH, max_to_keep=5)

try:
    load_weight()
    train(MODEL, LOSS, OPTIMIZER, CKPT_MANAGER)
except:
    print("Load weight after train")
    load_weight()



    
