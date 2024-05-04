import tkinter as tk
from tkinter import filedialog, colorchooser, Label, Entry, StringVar
from PIL import Image, ImageTk
import cv2
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage

seeds = []


def load_image():
    global img, img_tk, path
    path = filedialog.askopenfilename()
    img = Image.open(path)
    img_tk = ImageTk.PhotoImage(img)
    label_image.config(image=img_tk)
    label_image.image = img_tk


def apply_segmentation():
    global img_tk, seeds
    image_cv = cv2.imread(path)
    image_cv_lab = cv2.cvtColor(image_cv, cv2.COLOR_BGR2Lab)
    if color_space_var.get() == "RGB":
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

    if segment_type.get() == 1:
        color = colorchooser.askcolor(title="Choose color")[0]
        if color:
            color = [int(c) for c in color]
            image_cv = select_color(image_cv, color)
            image_cv_lab = select_color(image_cv_lab, color)

    # Flatten the image arrays
    pixels = image_cv.reshape((-1, 3))
    pixels_lab = image_cv_lab.reshape((-1, 3))

    if segmentation_method.get() == "KMeans":
        num_clusters = int(entry_clusters.get())
        clustering = KMeans(n_clusters=num_clusters)
        labels = clustering.fit_predict(pixels)
        clustering_lab = KMeans(n_clusters=num_clusters)
        labels_lab = clustering_lab.fit_predict(pixels_lab)
    elif segmentation_method.get() == "DBSCAN":
        eps = float(entry_eps.get())
        min_samples = int(entry_min_samples.get())
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clustering.fit_predict(pixels)
        clustering_lab = DBSCAN(eps=eps, min_samples=min_samples)
        labels_lab = clustering_lab.fit_predict(pixels_lab)
    elif seeds:
        output_image = grow_region(image_cv, seeds)
        output_image = Image.fromarray(output_image)  # Преобразуем обратно в PIL Image для отображения в Tkinter
        img_tk = ImageTk.PhotoImage(output_image)
        label_image.config(image=img_tk)
        label_image.image = img_tk

    # Generate colors for each label
    unique_labels = np.unique(labels)
    unique_labels_lab = np.unique(labels_lab)
    colors = [list(np.random.choice(range(256), size=3)) for _ in unique_labels]
    colors_lab = [list(np.random.choice(range(256), size=3)) for _ in unique_labels_lab]

    # Create segmented image arrays
    segmented_img = np.zeros((image_cv.shape[0], image_cv.shape[1], 3), dtype=np.uint8)
    segmented_img_lab = np.zeros((image_cv_lab.shape[0], image_cv_lab.shape[1], 3), dtype=np.uint8)

    # Map each pixel according to its label
    for label, color in zip(unique_labels, colors):
        segmented_img[labels.reshape(image_cv.shape[:2]) == label] = color
    for label, color in zip(unique_labels_lab, colors_lab):
        segmented_img_lab[labels_lab.reshape(image_cv_lab.shape[:2]) == label] = color

    # Convert array to image for displaying
    segmented_img = Image.fromarray(segmented_img)
    segmented_img_lab = Image.fromarray(segmented_img_lab)
    img_tk = ImageTk.PhotoImage(segmented_img)
    label_image.config(image=img_tk)
    label_image.image = img_tk

    # Display comparison using Matplotlib
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(segmented_img)
    plt.title('RGB Space')
    plt.subplot(1, 2, 2)
    plt.imshow(segmented_img_lab)
    plt.title('CIE Lab Space')
    plt.show()



def grow_region(image, seeds, threshold=30):
    # Преобразуем изображение в массив
    img_array = np.array(image)
    # Маска для отслеживания посещенных и добавленных пикселей
    mask = np.zeros(img_array.shape[:2], dtype=bool)
    output_image = np.zeros_like(img_array)

    for seed in seeds:
        x, y = seed
        seed_value = img_array[x, y]
        to_grow = [(x, y)]
        while to_grow:
            x, y = to_grow.pop(0)
            if mask[x, y]:
                continue
            mask[x, y] = True
            output_image[x, y] = seed_value
            # Проверка соседей
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < img_array.shape[0] and 0 <= ny < img_array.shape[1]:
                    if not mask[nx, ny]:
                        color_diff = np.linalg.norm(img_array[nx, ny] - seed_value)
                        if color_diff < threshold:
                            to_grow.append((nx, ny))
    return output_image


def select_color(image, color):
    lower = np.array(color) - 10
    upper = np.array(color) + 10
    mask = cv2.inRange(image, lower, upper)
    return cv2.bitwise_and(image, image, mask=mask)


def update_params(event):
    if segmentation_method.get() == "KMeans":
        label_clusters.pack()
        entry_clusters.pack()
        label_eps.pack_forget()
        entry_eps.pack_forget()
        label_min_samples.pack_forget()
        entry_min_samples.pack_forget()
    elif segmentation_method.get() == "DBSCAN":
        label_clusters.pack_forget()
        entry_clusters.pack_forget()
        label_eps.pack()
        entry_eps.pack()
        label_min_samples.pack()
        entry_min_samples.pack()


def on_image_click(event):
    x, y = event.x, event.y
    seed = (y, x)  # Обратите внимание на порядок, так как tkinter возвращает координаты в формате x, y
    if 'seeds' not in globals():
        global seeds
        seeds = []
    seeds.append(seed)
    apply_segmentation()  # Обновляем изображение каждый раз после добавления семени


root = tk.Tk()
root.title("Image Segmentation App")

btn_load = tk.Button(root, text="Load Image", command=load_image)
btn_load.pack()

segment_type = tk.IntVar()
rb_full = tk.Radiobutton(root, text="Full Segmentation", variable=segment_type, value=0)
rb_full.pack()
rb_color = tk.Radiobutton(root, text="Color Segmentation", variable=segment_type, value=1)
rb_color.pack()

segmentation_method = StringVar(value="KMeans")
segmentation_menu = tk.OptionMenu(root, segmentation_method, "KMeans", "DBSCAN", "Region Growing", "Watershed", command=update_params)
segmentation_menu.pack()

label_clusters = Label(root, text="Number of Clusters:")
entry_clusters = Entry(root)
entry_clusters.insert(0, "3")

label_eps = Label(root, text="EPS:")
entry_eps = Entry(root)
entry_eps.insert(0, "30")

label_min_samples = Label(root, text="Min Samples:")
entry_min_samples = Entry(root)
entry_min_samples.insert(0, "10")

color_space_var = StringVar(value="RGB")
option_menu = tk.OptionMenu(root, color_space_var, "RGB", "CIE Lab")
option_menu.pack()

btn_segment = tk.Button(root, text="Segment Image", command=apply_segmentation)
btn_segment.pack()

label_image = tk.Label(root)
label_image.pack()

label_image.bind("<Button-1>", on_image_click)  # Привязка события клика к изображению

root.mainloop()
