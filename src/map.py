import tkinter as tk
from tkinter import filedialog
from tkinter import ttk

from networkx.algorithms.distance_measures import radius
from tkintermapview import TkinterMapView
import time
import threading
import imageio
from PIL import Image, ImageTk
import math
from logic import analysis

video_reader = None
circle = None


def draw_circle_on_map(map_widget, lat, lon, radius_m=500, color_outline="blue", num_points=50):
    # radius_m - radius in meters
    # num_points - number of points approximating the circle

    # Earth's radius in meters
    R = 6378137

    points = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        # offset in meters
        dx = radius_m * math.cos(angle)
        dy = radius_m * math.sin(angle)

        # Offset coordinates in degrees
        dLat = dy / R * (180 / math.pi)
        dLon = dx / (R * math.cos(math.pi * lat / 180)) * (180 / math.pi)

        points.append((lat + dLat, lon + dLon))

    polygon = map_widget.set_polygon(points, outline_color=color_outline)
    return polygon

# video analysis function
def analyze_video(path, frame_skip):
    print(f"Analyzing file: {path}")
    return analysis.analyze_video(path, frame_skip)

def select_file():
    file_types = [("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
    file_path = filedialog.askopenfilename(title="Select a video file", filetypes=file_types)

    if file_path:
        label_file.config(text=f"Selected: {file_path}")
        start_analysis(file_path)
    else:
        label_file.config(text="No file selected")

def start_analysis(file_path):
    try:
        frame_skip = int(entry_frame_skip.get())
        if frame_skip < 1:
            frame_skip = 1
    except ValueError:
        frame_skip = 1

    progress_bar.pack(pady=10)
    progress_var.set(0)

    def run():
        cords = analyze_video(file_path, frame_skip)

        def after_analysis():
            show_map(cords[0][0], cords[0][1], cords[1])
            play_video(file_path)
            label_file.config(text=f"Analysis completed. Coordinates: {cords[0][0]}, {cords[0][1]}")
            progress_bar.pack_forget()

        window.after(0, after_analysis)

    threading.Thread(target=run).start()


def show_map(lat, lon, radius):
    global circle
    map_widget.pack(expand=True, fill="both")
    map_widget.set_position(lat, lon)
    map_widget.set_zoom(13)
    map_widget.set_marker(lat, lon, text="Warsaw")

    if circle:
        circle.delete()

    circle = draw_circle_on_map(map_widget, lat, lon, radius_m=radius)

def play_video(file_path):
    global video_reader
    video_label.pack(expand=True, fill="both")

    try:
        video_reader = imageio.get_reader(file_path)
        meta = video_reader.get_meta_data()
        fps = meta.get('fps', 25)
    except Exception as e:
        print("Error opening video:", e)
        return

    frame_iter = iter(video_reader)

    def stream():
        global video_reader
        try:
            frame = next(frame_iter)
            image = Image.fromarray(frame)
            image = image.resize((580, 360))
            photo = ImageTk.PhotoImage(image)
            video_label.config(image=photo)
            video_label.image = photo
            window.after(int(1000 / fps), stream)
        except StopIteration:
            print("Video finished.")
            if video_reader:
                video_reader.close()
                video_reader = None
        except Exception as e:
            print("Playback error:", e)
            if video_reader:
                video_reader.close()
                video_reader = None

    stream()



def reset_ui():
    global video_reader
    # Clear video
    video_label.config(image='')
    video_label.image = None
    video_label.pack_forget()

    # Clear map
    map_widget.set_position(50.0, 19.0)
    map_widget.set_zoom(6)
    map_widget.delete_all_marker()
    map_widget.pack_forget()

    # Reset progress
    progress_var.set(0)
    progress_bar.pack_forget()

    # Reset label and input
    label_file.config(text="No file selected")
    entry_frame_skip.delete(0, tk.END)
    entry_frame_skip.insert(0, "1")

    # Close video reader if open
    try:
        if video_reader:
            video_reader.close()
            video_reader = None
    except Exception as e:
        print("Error closing video reader:", e)


# ----------------------------

# ---------- MAIN WINDOW ----------
window = tk.Tk()
window.title("Video Analysis and Map")
window.geometry("1200x600")
window.configure(bg="#2e2e2e")

# ----------- TOP CONTROLS ----------

controls_frame = tk.Frame(window, bg="#2e2e2e")
controls_frame.pack(pady=10)

button_select = tk.Button(controls_frame, text="Select Video File", command=select_file,
                          bg="#444", fg="white", activebackground="#666", relief="flat")
button_select.pack(side="left", padx=5)

button_reset = tk.Button(controls_frame, text="Reset", command=lambda: reset_ui(),
                         bg="#444", fg="white", activebackground="#666", relief="flat")
button_reset.pack(side="left", padx=5)

label_file = tk.Label(window, text="No file selected", bg="#2e2e2e", fg="white")
label_file.pack(pady=5)

progress_var = tk.IntVar()
progress_bar = ttk.Progressbar(window, orient="horizontal", length=400, mode="determinate", variable=progress_var)

# ----------- FRAME INTERVAL ENTRY ----------
frame_input_frame = tk.Frame(window, bg="#2e2e2e")
frame_input_frame.pack(pady=(10, 0))

label_frame_skip = tk.Label(frame_input_frame, text="Analyze every Nth frame:", bg="#2e2e2e", fg="white")
label_frame_skip.pack(side="left", padx=(0, 5))

entry_frame_skip = tk.Entry(frame_input_frame, width=5)
entry_frame_skip.insert(0, "1")  # analyze every frame by default
entry_frame_skip.pack(side="left")

# ----------- MAIN CONTENT AREA ----------
content_frame = tk.Frame(window, bg="#2e2e2e")
content_frame.pack(fill="both", expand=True, padx=10, pady=10)

# LEFT: MAP FRAME
map_frame = tk.Frame(content_frame, bg="#2e2e2e")
map_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

map_widget = TkinterMapView(map_frame, width=580, height=480, corner_radius=10)

# RIGHT: VIDEO FRAME
video_frame = tk.Frame(content_frame, bg="#2e2e2e")
video_frame.pack(side="right", fill="both", expand=True)

video_label = tk.Label(video_frame, bg="#2e2e2e")

window.mainloop()
