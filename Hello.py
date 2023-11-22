# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit.logger import get_logger
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import onnxruntime as ort
from skimage.transform import resize
from ultralytics import YOLO
from time import time

# https://blog.streamlit.io/introducing-new-layout-options-for-streamlit/
# https://docs.streamlit.io/library/api-reference/widgets/st.file_uploader

# https://docs.streamlit.io/library/api-reference/widgets/st.download_button
#scroll for image download

# Terminal: streamlit run /workspaces/dentalsegmentator/Hello.py
LOGGER = get_logger(__name__)

def sigmoid(x):
   return 1/(1+np.exp(-x))


@st.cache_resource
def GetYolo():
    return YOLO(os.path.join('onnx','yolo.pt'))

@st.cache_resource
def GetOnnxSession():
    onnx_path = os.path.join(os.curdir,'onnx','model.onnx')
    ort_sess = ort.InferenceSession(onnx_path,providers = ['CPUExecutionProvider'])
    return ort_sess
    

def Predict_Yolo(im_path,yolo):
    # yolo = YOLO(os.path.join('onnx','yolo.pt'))  # pretrained YOLOv8n model

    im_path = r'blob.png'
    res = yolo(im_path)[0]    
    plt.imsave('yolo_output.png',res.plot(line_width = 3,font_size = None))
    
    
def Predict(img,ort_sess):
   print('In prediction')
   # onnx_path = os.path.join(os.curdir,'onnx','model.onnx')
   # ort_sess = ort.InferenceSession(onnx_path,providers = ['CPUExecutionProvider'])
   init_shape = img.shape
   if img.max()>1:
       img =img/255.
   if img.shape[0] ==3:
       img = img[0]
   elif img.shape[2]==3:
       img = img[:,:,0]
   dummy_input = resize(img,(512,512),anti_aliasing=True,preserve_range=True).astype(np.float32)
   dummy_input = np.expand_dims(dummy_input, axis = 0)
   dummy_input = np.expand_dims(dummy_input, axis = 0)
   outputs = ort_sess.run(None, {'input': dummy_input})[0].squeeze() #ort_sess.get_inputs()[0]
   outputs = 255*(sigmoid(outputs)>.5)
   outputs = resize(outputs,init_shape,order=0)
   return outputs.astype(np.uint8)

def run():
    st.set_page_config(
        page_title="Dental Segmentator",
        page_icon="🦷",
        layout='wide'
    )
    
    
    yolo = GetYolo()
    ort_sess = GetOnnxSession()
    
    st.write("#                🦷 Welcome to Dental Segmentator! 🦷")
    st.write("## A demo web application for image segmentation in dentistry")
    st.write("### Author: Emmanuel Montagnon (21-11-2023)")
    # st.write(f"currdir {os.path.abspath(os.curdir)}")
    # st.write(f"OnnxPath {os.path.exists(os.path.join(os.curdir,'onnx','model.onnx'))}")
    # st.sidebar.success("Select a demo above.")
    files = st.file_uploader(label='Please load an image',accept_multiple_files=False,
                     on_change=None)
    
    image = Image.open(files)
    img_array = np.array(image)
    plt.imsave('blob.png',img_array)

    # st.write(f'Input image shape: {img_array.shape} min: {img_array.min()} max: {img_array.max()}')
    
    
    t0 = time()
    output = Predict(img_array,ort_sess)
    semantic_time = time()-t0
    # f,ax = plt.subplots()
    # ax.imshow(img_array)
    # print(.5*(output/255))
    # ax.imshow(output,alpha = .5)
    # f.savefig('output.png')
    # output = plt.imread('output.png')

    plt.imsave('output.png',output,cmap = 'gray')
    
    
    t0 = time()
    Predict_Yolo('blob.png',yolo)
    yolo_time = time()-t0
    img_array = plt.imread('yolo_output.png',output)
    
    
    
    
    
    col1,col2,col3 = st.columns(3)
    col1.header("Input image\n Panoramix Xray")
    col2.header(f"Semantic segmentation\nCPU inference time: {semantic_time:.2f} sec")
    col3.header(f"Instance segmentation\nCPU inference time: {yolo_time:.2f} sec")
    col1.image('blob.png')
    col2.image('output.png')
    col3.image('yolo_output.png')
    
    
    st.write('## Input image')
    img_array = plt.imread('blob.png')
    st.image(image=img_array)
    # vv = st.selectbox(label='Pick an image',options = ['a','b','c'])
    # st.write(vv)
    st.write('## Semantic segmentation output')
    st.write(f"CPU inference time: {semantic_time:.2f} sec")
    st.image(image=output)
    
    st.write('## Instance segmentation output')
    st.write(f"CPU inference time: {yolo_time:.2f} sec")
    st.image(image='yolo_output.png')
    
    
    with open("output.png", "rb") as file:
      btn = st.download_button(
              label="Download image",
              data=file,
              file_name="blob.png",
              mime="image/png"
            )
      
      
    # st.markdown(
    #     """
    #     Streamlit is an open-source app framework built specifically for
    #     Machine Learning and Data Science projects.
    #     **👈 Select a demo from the sidebar** to see some examples
    #     of what Streamlit can do!
    #     ### Want to learn more?
    #     - Check out [streamlit.io](https://streamlit.io)
    #     - Jump into our [documentation](https://docs.streamlit.io)
    #     - Ask a question in our [community
    #       forums](https://discuss.streamlit.io)
    #     ### See more complex demos
    #     - Use a neural net to [analyze the Udacity Self-driving Car Image
    #       Dataset](https://github.com/streamlit/demo-self-driving)
    #     - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
    # """
    # )


if __name__ == "__main__":
    run()
