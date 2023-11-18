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


# https://docs.streamlit.io/library/api-reference/widgets/st.download_button
#scroll for image download


LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )

    st.write("# Welcome to Manututu! 👋")
    st.write(f"currdir {os.path.abspath(os.curdir)}")
    st.sidebar.success("Select a demo above.")
    files = st.file_uploader(label='coucou',accept_multiple_files=True,
                     on_change=None)
    
    image = Image.open(files[0])
    img_array = np.array(image)
    plt.imsave('blob.png',img_array)
    st.write(files)
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
