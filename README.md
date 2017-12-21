# Visual-Tracking-using-Background-Subtraction
The goal of this Visual Tracking module is to learn about, and, more importantly, to learn how to use basic tracking algorithms and evaluate their performances. We will start with a very simple and effective technique called Background Subtraction which can be used to initialize the tracker, i.e. to find the target’s position in the first frame of a sequence, or to track the target through the entire sequence. Background Subtraction Background subtraction (BS) is widely used in surveillance and security applications, and serves as a first step in detecting objects or people in videos. BS is based on a model of the scene background that is the static part of the scene. Each pixel is analyzed and a deviation from the model is used to classify pixels as being background or foreground.