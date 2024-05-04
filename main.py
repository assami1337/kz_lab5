import tkinter as tk
from tkinter import filedialog, colorchooser, Label, Entry, StringVar
from PIL import Image, ImageTk
import cv2
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from skimage import measure, filters, color
from skimage.segmentation import watershed
from scipy import ndimage as ndi

seeds = []


def generate_fixed_colors(n_colors):
    np.random.seed(42)
    return [list(np.random.choice(range(256), size=3)) for _ in range(n_colors)]


def load_image():
    global img, img_tk, path
    path = filedialog.askopenfilename()
    img = Image.open(path)
    img_tk = ImageTk.PhotoImage(img)
    label_image.config(image=img_tk)
    label_image.image = img_tk


def apply_segmentation():
    global img_tk, initial_seed, initial_region
    image_cv = cv2.imread(path)
    image_cv_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    image_cv_lab = cv2.cvtColor(image_cv, cv2.COLOR_BGR2Lab)

    if segment_type.get() == 1:
        color = colorchooser.askcolor(title="Choose color")[0]
        if color:
            color = [int(c) for c in color]
            image_cv_rgb = select_color(image_cv_rgb, color)
            image_cv_lab = select_color(image_cv_lab, color)

    pixels_rgb = image_cv_rgb.reshape((-1, 3))
    pixels_lab = image_cv_lab.reshape((-1, 3))

    if segmentation_method.get() == "KMeans":
        num_clusters = int(entry_clusters.get())
        clustering_rgb = KMeans(n_clusters=num_clusters)
        labels_rgb = clustering_rgb.fit_predict(pixels_rgb)
        clustering_lab = KMeans(n_clusters=num_clusters)
        labels_lab = clustering_lab.fit_predict(pixels_lab)
        show_comparison(image_cv_rgb, labels_rgb, image_cv_lab, labels_lab)
    elif segmentation_method.get() == "DBSCAN":
        eps = float(entry_eps.get())
        min_samples = int(entry_min_samples.get())
        clustering_rgb = DBSCAN(eps=eps, min_samples=min_samples)
        labels_rgb = clustering_rgb.fit_predict(pixels_rgb)
        clustering_lab = DBSCAN(eps=eps, min_samples=min_samples)
        labels_lab = clustering_lab.fit_predict(pixels_lab)
        show_comparison(image_cv_rgb, labels_rgb, image_cv_lab, labels_lab)
    elif segmentation_method.get() == "Seed Growing":
        output_image = grow_region(image_cv_rgb, initial_seed)
        img_pil = Image.fromarray(output_image)
        img_tk = ImageTk.PhotoImage(img_pil)
        label_image.config(image=img_tk)
        label_image.image = img_tk
    elif segmentation_method.get() == "Region Growing":
        output_image = grow_region_from_region(image_cv_rgb, initial_region)
        img_pil = Image.fromarray(output_image)
        img_tk = ImageTk.PhotoImage(img_pil)
        label_image.config(image=img_tk)
        label_image.image = img_tk
    elif segmentation_method.get() == "Watershed":
        output_image = apply_watershed(image_cv_rgb)
        img_pil = Image.fromarray((output_image * 255).astype(np.uint8))
        img_tk = ImageTk.PhotoImage(img_pil)
        label_image.config(image=img_tk)
        label_image.image = img_tk


def show_comparison(image_rgb, labels_rgb, image_lab, labels_lab):
    unique_labels_rgb = np.unique(labels_rgb)
    unique_labels_lab = np.unique(labels_lab)
    colors_rgb = [list(np.random.choice(range(256), size=3)) for _ in unique_labels_rgb]
    colors_lab = [list(np.random.choice(range(256), size=3)) for _ in unique_labels_lab]

    segmented_img_rgb = np.zeros((image_rgb.shape[0], image_rgb.shape[1], 3), dtype=np.uint8)
    segmented_img_lab = np.zeros((image_lab.shape[0], image_lab.shape[1], 3), dtype=np.uint8)

    for label, color in zip(unique_labels_rgb, colors_rgb):
        segmented_img_rgb[labels_rgb.reshape(image_rgb.shape[:2]) == label] = color
    for label, color in zip(unique_labels_lab, colors_lab):
        segmented_img_lab[labels_lab.reshape(image_lab.shape[:2]) == label] = color

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(segmented_img_rgb)
    plt.title('RGB Space')
    plt.subplot(1, 2, 2)
    plt.imshow(segmented_img_lab)
    plt.title('CIE Lab Space')
    plt.show()


def show_dbscan_variations():
    image_cv = cv2.imread(path)
    image_cv_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    pixels = image_cv_rgb.reshape((-1, 3))

    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 8))
    axes = axes.flatten()

    parameters = [(1, 4), (2, 4), (3, 4), (4, 4)]
    fixed_colors = generate_fixed_colors(256)

    for ax, (eps, min_samples) in zip(axes, parameters):
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clustering.fit_predict(pixels)
        unique_labels = np.unique(labels)

        segmented_img = np.zeros((image_cv_rgb.shape[0], image_cv_rgb.shape[1], 3), dtype=np.uint8)
        for label in unique_labels:
            color = fixed_colors[label % len(fixed_colors)]
            segmented_img[labels.reshape(image_cv_rgb.shape[:2]) == label] = color

        ax.imshow(segmented_img)
        ax.set_title(f"EPS: {eps}, Min Samples: {min_samples}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def grow_region(image, initial_seed):
    img_array = np.array(image)
    mask = np.zeros(img_array.shape[:2], dtype=bool)
    output_image = np.zeros_like(img_array)

    def region_grow(x, y):
        seed_value = img_array[x, y]
        to_grow = [(x, y)]
        while to_grow:
            x, y = to_grow.pop(0)
            if mask[x, y]:
                continue
            mask[x, y] = True
            output_image[x, y] = seed_value
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < img_array.shape[0] and 0 <= ny < img_array.shape[1]:
                    if not mask[nx, ny]:
                        color_diff = np.linalg.norm(img_array[nx, ny] - seed_value)
                        if color_diff < 220:
                            to_grow.append((nx, ny))

    # Начать сегментацию из первого выбранного семени
    x, y = initial_seed
    if not mask[x, y]:
        region_grow(x, y)

    # Продолжить сегментацию из всех непосещенных пикселей
    for x in range(img_array.shape[0]):
        for y in range(img_array.shape[1]):
            if not mask[x, y]:
                region_grow(x, y)

    return output_image


def grow_region_from_region(image, initial_region):
    img_array = np.array(image)
    mask = np.zeros(img_array.shape[:2], dtype=bool)
    output_image = np.zeros_like(img_array)

    start, end = initial_region

    def region_grow(x, y):
        seed_value = img_array[x, y]
        to_grow = [(x, y)]
        while to_grow:
            x, y = to_grow.pop(0)
            if mask[x, y]:
                continue
            mask[x, y] = True
            output_image[x, y] = seed_value
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < img_array.shape[0] and 0 <= ny < img_array.shape[1]:
                    if not mask[nx, ny]:
                        color_diff = np.linalg.norm(img_array[nx, ny] - seed_value)
                        if color_diff < 220:
                            to_grow.append((nx, ny))

    # Инициировать рост из всех пикселей в начальном регионе
    for i in range(start[0], end[0]):
        for j in range(start[1], end[1]):
            if not mask[i, j]:
                region_grow(i, j)

    # Продолжить сегментацию из всех непосещенных пикселей
    for x in range(img_array.shape[0]):
        for y in range(img_array.shape[1]):
            if not mask[x, y]:
                region_grow(x, y)

    return output_image


def apply_gaussian_blur(image, kernel_size=5):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def apply_watershed(image):
    smoothed = apply_gaussian_blur(image)

    gray = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)

    gradient = filters.sobel(gray)

    markers = ndi.label(gradient < filters.threshold_otsu(gradient))[0]

    labels = watershed(gradient, markers)

    segmented_image = color.label2rgb(labels, image, kind='overlay')

    return segmented_image


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
    global initial_seed, rect_start, initial_region
    if segmentation_method.get() == "Seed Growing":
        x, y = event.x, event.y
        initial_seed = (y, x)
        apply_segmentation()
    elif segmentation_method.get() == "Region Growing":
        if 'rect_start' not in globals():
            rect_start = (event.y, event.x)
        else:
            rect_end = (event.y, event.x)
            initial_region = (rect_start, rect_end)
            apply_segmentation()
            del rect_start


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
segmentation_menu = tk.OptionMenu(root, segmentation_method, "KMeans", "DBSCAN", "Seed Growing", "Region Growing",
                                  "Watershed",
                                  command=update_params)
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

btn_show_variations = tk.Button(root, text="Show DBSCAN Variations", command=show_dbscan_variations)
btn_show_variations.pack()

label_image = tk.Label(root)
label_image.pack()

label_image.bind("<Button-1>", on_image_click)

root.mainloop()
