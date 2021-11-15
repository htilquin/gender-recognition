import streamlit as st
from utils_app import *

st.set_page_config(
    page_title="Face Detection", 
    page_icon="😃", 
    layout='centered', 
    initial_sidebar_state='auto', 
    menu_items={'About': """### Face Detector 
    \nThis app has been created to help you get the faces out of any picture.
    \n ---
    \n Face detection : HOG based detector from Dlib.
    \n Shape predictor : 68 face landmarks (Dlib).
    \n Face recognition model : resnet model v1 (Dlib).
    \n Gender classification : Multi-Layer Perceptron."""})

start_timer = start_timer()

### SIDEBAR
st.sidebar.markdown("### Options")
gender_rec = st.sidebar.checkbox("Gender recognition")
if gender_rec :
    parity = st.sidebar.checkbox("Gender-Parity Meter")
    cols = st.sidebar.columns(3)
    hex_color_female = cols[0].color_picker("Female-frame", value="#FC8BFF")
    picked_color_female = hex2RBG(hex_color_female)
    hex_color_male = cols[1].color_picker("Male-frame", value="#7F76FF")
    picked_color_male = hex2RBG(hex_color_male)
else :
    hex_color = st.sidebar.color_picker("Frame color", value="#FF0000")
    picked_color = hex2RBG(hex_color)
frame_weight = st.sidebar.number_input("Frame weight", min_value=1, max_value=8, value=3)
number_of_cols = st.sidebar.slider("Results grid-size", min_value=1, max_value=8, value=6)

st.sidebar.markdown("### File")
uploaded_picture = st.sidebar.file_uploader("Choose a picture", type=['png', 'jpg', 'jpeg'], )


st.markdown("## Face Detection")
if uploaded_picture is None :
    st.write("⟵ You can upload a picture using the sidebar :)")


else :
    # get all the faces from the picture
    detector = get_face_detector()

    image = read_image_from_streamlit_cv2(uploaded_picture)
    image_clean = image.copy()
    faces = detect_faces(image, detector)

    if len(faces) == 0 :
        caption = "No face detected..."
        st.image(image, caption)

    elif gender_rec :
        female_images = []
        male_images = []

        # load config and model
        shape_pred = get_shape_predictor()
        face_rec_model = get_face_rec_model()
        clf_model = load_trained_classifier()

    else :
        face_images = []

    for face_rect in faces :
        # make rectangle wider
        wider_face_rect = get_wider_rectangle(face_rect)
        face_image = image_clean[wider_face_rect.top(): wider_face_rect.bottom(), wider_face_rect.left(): wider_face_rect.right()].copy()

        if gender_rec :
            # calculate features
            face_embeddings = get_face_embeddings(image_clean, wider_face_rect, shape_pred, face_rec_model)
            # predicts
            prediction = get_gender_from_features(clf_model, face_embeddings)
            # add picture to the accumulator
            if prediction == "female" :
                female_images.append(face_image)
                add_rectangle_on_image(image, wider_face_rect, picked_color_female, frame_weight)
            else :
                male_images.append(face_image)
                add_rectangle_on_image(image, wider_face_rect, picked_color_male, frame_weight)

        else :
            face_images.append(face_image)
            # Ajoute un rectangle coloré sur la face
            add_rectangle_on_image(image, wider_face_rect, picked_color, frame_weight)


    # DISPLAY PICTURES
    if gender_rec and len(faces) > 0 :
        pc_women = len(female_images)/(len(female_images)+ len(male_images))
        caption = f"{len(faces)} faces detected, {pc_women*100:.0f}% of which are women."

        st.image(image, caption)

        # DISPLAY GENDER PARITOMETER
        if parity :
            cols = st.columns((1,3,1))
            cols[1].pyplot(display_paritometer(pc_women, hex_color_female, hex_color_male))

        # display all "female" pictures
        if len(female_images) > 0 :
            st.write(f"Female, {len(female_images)}")      
            grid_display(female_images, number_of_cols)

        # display all "male" pictures
        if len(male_images) > 0 :
            st.write(f"Male, {len(male_images)}")
            grid_display(male_images, number_of_cols)

    elif len(faces) > 0 :
        caption = f"{len(faces)} faces detected."
        st.image(image, caption)

        grid_display(face_images, number_of_cols)


st.markdown(footer,unsafe_allow_html=True)

st.write(f"Total loading time : {stop_timer(start_timer)} sec.")
