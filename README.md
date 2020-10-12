# smomi3


## flip
```
def augment(image,label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.random_flip_left_right(image)
    return image,label
 ```   
    
    
## brightness/contrast
```
def augment(image,label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.random_brightness(image, 0.7, seed=None)
    image = tf.image.random_contrast(image, 0.3, 1.3, seed=None)
    return image,label
 ```
    
    
## noize
```
def augment(image,label):
    with tf.name_scope('Add_gaussian_noise'):
        noise_img = image + tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=1.0, dtype=tf.float32)
        noise_img = tf.clip_by_value(noise_img, -1.0, 1.0)
    return noise_img,label
```
   
## rotate
```
def augment(image,label):
    image = tfa.image.rotate(image, 60)
    return image,label
```   

## optional
```

```   
   
    
    
    
