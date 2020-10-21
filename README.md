# smomi3


## Flip
```
def augment(image,label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.random_flip_left_right(image)
    return image,label
 ``` 
train - оранжевый
val - синий

![alt text](https://github.com/Uniderwy/smomi3/blob/main/new_flip.jpg) 
    
    
## Brightness/Contrast
```
def augment(image,label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.random_brightness(image, 0.7, seed=None)
    image = tf.image.random_contrast(image, 0.3, 1.3, seed=None)
    return image,label
 ```
### Brightness 0.1, Contract 0.7, 1.4
train - зеленый
val - серый

![alt text](https://github.com/Uniderwy/smomi3/blob/main/new_bright4.jpg) 

### Brightness 0.1, Contract 0.2, 1.5
train - красный 
val - голубой

![alt text](https://github.com/Uniderwy/smomi3/blob/main/new_bright2.jpg) 

### Brightness 0.7, Contract 0.3, 1.3
train - оранжевый
val - синий

![alt text](https://github.com/Uniderwy/smomi3/blob/main/new_bright3.jpg) 
    
    
## Gauss noize
```
def augment(image,label):
    with tf.name_scope('Add_gaussian_noise'):
        noise_img = image + tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.5, dtype=tf.float32)
        noise_img = tf.clip_by_value(noise_img, -1.0, 1.0)
    return noise_img,label
```
### Gauss stddev = 0.5
train - красный
val - голубой

![alt text](https://github.com/Uniderwy/smomi3/blob/main/gauss2.jpg)    
   
### Gauss stddev = 0.25
train - розовый
val - зеленый

![alt text](https://github.com/Uniderwy/smomi3/blob/main/gauss4.jpg)  

### Gauss stddev = 0.1
train - серый
val - оранжевый

![alt text](https://github.com/Uniderwy/smomi3/blob/main/gauss6.jpg)    


## Rotate
```
def augment(image,label):
    image = tfa.image.rotate(image, 60)
    return image,label
```   
### Rotate60
train - серый
val - оранжевый

![alt text](https://github.com/Uniderwy/smomi3/blob/main/rotate60.jpg)  

### Rotate45
train - синий
val - красный

![alt text](https://github.com/Uniderwy/smomi3/blob/main/rotate45.jpg) 

### Rotate30
train - красный
val - голубой

![alt text](https://github.com/Uniderwy/smomi3/blob/main/rotate30.jpg) 


## Optional
```      
def augment(image,label):
    with tf.name_scope('Add_gaussian_noise'):
        image = image + tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.25, dtype=tf.float32)
        image = tf.clip_by_value(image, -1.0, 1.0)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.1, seed=None)
    image = tf.image.random_contrast(image, 0.7, 1.4, seed=None)
    image = tfa.image.rotate(image, 30)
    return image,label
```   
train - синий
val - красный

![alt text](https://github.com/Uniderwy/smomi3/blob/main/optimal1.jpg) 
  

  
  
    
    
    
