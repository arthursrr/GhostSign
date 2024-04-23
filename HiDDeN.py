from typing import Any, Tuple

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.layers import (Activation, BatchNormalization,
                                     Concatenate, Conv2D, Conv2DTranspose,
                                     Dense, GaussianNoise,
                                     GlobalAveragePooling2D, Input, MaxPool2D)
from tensorflow.keras.losses import BinaryCrossentropy


class HiDDeN:
    def __init__(self, 
        img_shape: Tuple[int, int, int] = (128, 128, 3), 
        msg_shape: Tuple[int, int, int] = ( 16, 16, 1), 
        kernel_size: int = 3,
    ) -> None:
        self.initializer = GlorotNormal(seed=0)
        self.img_shape = img_shape
        self.msg_shape = msg_shape
        self.kernel_size = kernel_size
        self.msg_length = 256


    @staticmethod
    def discriminator_loss(disc_real_output, disc_generated_output):
        loss_object = BinaryCrossentropy(from_logits=True)
        
        real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

        generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss


    @staticmethod
    def generator_loss(disc_generated_output, gen_output, target, reader_output, msg_target):
        loss_object = BinaryCrossentropy(from_logits=True)
        
        gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
        reader_loss = loss_object(tf.ones_like(reader_output), msg_target)
        
        # Mean absolute error
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

        total_gen_loss = gan_loss + reader_loss + (100 * l1_loss)

        return total_gen_loss, gan_loss, reader_loss, l1_loss


    def Conv2DBlock(
        self, 
        filters: int, 
        kernel_size: int, 
        strides: int = 1, 
        name: str ='ConvBlock'
    ) -> Sequential:
        block = Sequential(name=f'{name}_{filters}')
        block.add(
            Conv2D(
                filters, 
                kernel_size, 
                strides=strides, 
                padding='same', 
                kernel_initializer=self.initializer,
            )
        )
        block.add(BatchNormalization())
        block.add(Activation("relu"))
        return block


    def Pool_EncodeBlock(self, input: Any, filters: int, kernel_size: int) -> Any:
        x = Conv2D(
            filters, 
            kernel_size, 
            strides = 1, 
            padding='same', 
            kernel_initializer = self.initializer, 
            name=f'Encode_Conv_{filters}'
        )(input)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        pool = MaxPool2D(
            pool_size=(2, 2), 
            strides=2, 
            padding='same',
            name=f"Encode_MaxPool_{filters}",
        )(x)
        return x, pool


    def Stride_EncodeBlock(
        self, 
        filters: int, 
        kernel_size: int, 
        name: str='Encode_ConvBlock'
    ) -> Sequential:
        block = Sequential(name=f'{name}_{filters}')
        block.add(
            self.Conv2DBlock(
                filters, 
                kernel_size, 
                strides=1, 
                name=name,
            ),
        )
        block.add(
            self.Conv2DBlock(
                filters, 
                kernel_size, 
                strides=2, 
                name='Encode_Stides',
            ),
        )
        return block


    def DecodeBlock(
        self, 
        filters: int, 
        kernel_size: int, 
        name: str='Decode_ConvBlock',
    ) -> Sequential:
        block = Sequential(name=f'{name}_{filters}')
        block.add(
            Conv2DTranspose(
                filters, 
                kernel_size, 
                activation='relu', 
                strides=2, 
                padding='same', 
                kernel_initializer=self.initializer,
                name=f"Up_Conv_{filters}",
            )
        )
        block.add(
           self.Conv2DBlock(
                filters, 
                kernel_size, 
                strides=2, 
                name=name,
            ),
        )
        return block


    def Generator(self) -> Model:
        img_input = Input(shape=self.img_shape, name="img_input")
        msg_input = Input(shape=self.msg_shape, name="msg_input")
 
        msg_down_stack = [
            self.Conv2DBlock(256, self.kernel_size, name="Encode_ConvBlock_1"),  # (batch_size, 8, 8, 256)
            self.Stride_EncodeBlock(256, self.kernel_size, name="Encode_ConvBlock_2"),  # (batch_size, 8, 8, 256)
            self.Conv2DBlock(512, self.kernel_size, name="Encode_ConvBlock_3"),  # (batch_size, 4, 4, 512)
            self.Stride_EncodeBlock(512, self.kernel_size, name="Encode_ConvBlock_4"),  # (batch_size, 4, 4, 512)
            self.Conv2DBlock(1024, self.kernel_size, name='Final_Encode_Conv'), # (batch_size, 4, 4, 1024)
        ]

        last_conv = self.Conv2DBlock(3, self.kernel_size, name='Final_Decode_Conv')

        x=img_input
        y=msg_input
        
        # Downsampling through the model
        filters = [32, 64, 128, 256, 512]
        skips = []
        for filter in filters:
            s,x = self.Pool_EncodeBlock(x, filter, self.kernel_size)
            skips.append(s)  
        
        # (batch_size, 4, 4, 1024)
        x = self.Conv2DBlock(1024, self.kernel_size, name='Final_Encode_MSG_Conv')(x)
        
        skips.reverse()
        filters.reverse()
        
        for down in msg_down_stack:
            y = down(y)
        
        # FusionFeature step
        x = Concatenate(axis=3)([x, y])
        x = GaussianNoise(1)(x)
        x = self.Conv2DBlock(1024, self.kernel_size, name='Decode_Conv')(x)
        
        # Upsampling and establishing the skip connections
        for skip, filter in zip(skips, filters):
            x = Conv2DTranspose(
                filter, 
                self.kernel_size, 
                activation='relu', 
                strides=2, 
                padding='same', 
                kernel_initializer=self.initializer, 
                name=f"Decode_Up_Conv_{filter}"
            )(x)
            x = Concatenate(axis=3)([x, skip])
            x = self.Conv2DBlock(filter, 3, name='Decode_Conv')(x)
        
        output = last_conv(x)
        
        return Model(inputs=[img_input, msg_input], outputs=output)       


    def Reader(self) -> Model:
        inp = Input(shape=self.img_shape, name='input_image')
        x = inp
        # Applying 7 Conv-BN-ReLU blocks with 64 output filters
        for filters in [64, 64, 64, 64, 64, 64, 64]:
            x = Conv2D(
                filters,
                kernel_size=self.kernel_size,
                strides=1,
                padding='same',
                use_bias=False,
            )(x)
            x = BatchNormalization(axis=-1)(x)
            x = Activation("relu")(x)

        # Last ConvBNReLU with L filters
        x = Conv2D(
            self.msg_length,
            kernel_size=self.kernel_size,
            padding='same',
            use_bias=False,
        )(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        # Average Pooling over all spatial dimensions
        x = GlobalAveragePooling2D()(x)
        # Final linear layer with L units
        x = Dense(self.msg_length)(x)
        
        decoder_model = Model(inp, x, name='decoder')
        return decoder_model


    def Discriminator(self) -> Model:
        # build the adversary
        input_images = Input(shape=self.img_shape, name='generated_input')
        target_images = Input(shape=self.img_shape, name='target_input')
        
        x=Concatenate(axis=-1)([input_images, target_images])
        # Applying 3 Conv-BN-ReLU blocks with 64 output filters
        for filters in [64, 64, 64]:
            x = Conv2D(filters,
                       kernel_size=self.kernel_size,
                       strides=1,
                       padding='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Activation("relu")(x)
        # Average Pooling over all spatial dimensions
        x = GlobalAveragePooling2D()(x)
        # Final linear layer to classify the image
        adversary_output = Dense(2, activation="softmax")(x)
        discriminator_model = Model(
            input_images, adversary_output, name='discriminator')
        return discriminator_model