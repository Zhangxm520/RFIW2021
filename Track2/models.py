from keras.models import load_model,Model
from keras.layers import Input, Concatenate,Flatten
from keras.initializers import RandomUniform
from keras.optimizers import SGD
from Track2.losses import *


uniform = RandomUniform(minval=-0.05, maxval=0.05, seed=100)


def get_encoder(name):
    input1=Input((112,112,3))
    model = load_model('ArcFace_r100_v1.h5')
    x=model(input1)
    return Model(input1, x, name=name)


def get_projecter(input_shape,hidden_shape,out_shape,name):
    input1=Input((input_shape,))
    x = tf.keras.layers.Dense(units=hidden_shape,kernel_initializer=uniform)(input1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x)
    x = tf.keras.layers.Dense(units=out_shape, kernel_initializer=uniform)(x)
    return Model(input1, x, name=name)


def get_model():
    father_input = Input((112, 112, 3))
    mother_input = Input((112, 112, 3))
    child_input = Input((112, 112, 3))

    encoder=get_encoder("encoder")
    father_em, mother_em,child_em = encoder(father_input), encoder(mother_input),encoder(child_input)
    father_em=Flatten(name="father")(father_em)
    mother_em = Flatten(name="mother")(mother_em)
    child_em = Flatten(name="child")(child_em)

    projecter = get_projecter(input_shape=encoder.output.shape[-1],
                              hidden_shape=256,out_shape=128,name="projection")
    father_pro, mother_pro,child_pro = projecter(father_em), projecter(mother_em),projecter(child_em)
    out = Concatenate(axis=1)([father_pro, mother_pro,child_pro])
    model = Model([father_input, mother_input,child_input], out)
    model.compile(loss=compute2_loss, optimizer=SGD(1e-4, momentum=0.9))
    return model


def get_save_model(model):
    save_model=Model(model.input,
                     [model.get_layer('father').output,
                        model.get_layer('mother').output,
                      model.get_layer("child").output])
    save_model.compile()
    return save_model