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
import torch
import matplotlib.pyplot as plt
import onnxruntime as ort
from skimage.transform import resize
# https://docs.streamlit.io/library/api-reference/widgets/st.download_button
#scroll for image download

# Terminal: streamlit run /workspaces/dentalsegmentator/Hello.py
LOGGER = get_logger(__name__)

def sigmoid(x):
   return 1/(1+np.exp(-x))


@st.cache
def GetOrtSession():
   ort_sess = ort.InferenceSession('./onnx/model.onnx',providers = ['CPUExecutionProvider'])
   return ort_sess


def Predict(img,ort_sess):
   init_shape = img.shape
   dummy_input = resize(img,(512,512),anti_aliasing=True,preserve_range=True)
   outputs = ort_sess.run(None, {'input': dummy_input})[0].squeeze() #ort_sess.get_inputs()[0]
   outputs = 255*(sigmoid(outputs)>.5)
   return outputs.astype(np.int16)

def run():
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )
    #ort_sess = GetOrtSession()
    st.write("# Welcome to Manututu! 👋")
    st.write(f"currdir {os.path.abspath(os.curdir)}")
    st.sidebar.success("Select a demo above.")
    files = st.file_uploader(label='coucou',accept_multiple_files=False,
                     on_change=None)
    
    image = Image.open(files)
    img_array = np.array(image)

    st.write(f'Input image shape: {img_array.shape}')
    plt.imsave('blob.png',img_array)
    st.image(image=img_array)


    with open("blob.png", "rb") as file:
      btn = st.download_button(
              label="Download image",
              data=file,
              file_name="blob.png",
              mime="image/png"
            )
    st.markdown(
        """
        Streamlit is an open-source app framework built specifically for
        Machine Learning and Data Science projects.
        **👈 Select a demo from the sidebar** to see some examples
        of what Streamlit can do!
        ### Want to learn more?
        - Check out [streamlit.io](https://streamlit.io)
        - Jump into our [documentation](https://docs.streamlit.io)
        - Ask a question in our [community
          forums](https://discuss.streamlit.io)
        ### See more complex demos
        - Use a neural net to [analyze the Udacity Self-driving Car Image
          Dataset](https://github.com/streamlit/demo-self-driving)
        - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
    """
    )


if __name__ == "__main__":
    run()
