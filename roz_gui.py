import tkinter as tk
from PIL import ImageTk, Image
import tkFileDialog as filedialog
from roz import *

LARGE_FONT = ("Verdana", 12)

dataset_path = ''
img_path = ''
preds = []
w = None


def browse_image(event):
    global img_path
    img_path = filedialog.askopenfilename(filetypes=(("All files", "*.type"), ("All files", "*")))
    img = Image.open(img_path)
    # img = img.resize((700, 700))
    img.show()


def Browse_Folder(event):
    # Allow user to select a directory and store it in global var
    # called folder_path
    global dataset_path
    dataset_path = filedialog.askdirectory()
    print dataset_path


def extract_embeddings_function(event):
    print('extract_embeddings_button clicked')
    print(dataset_path)
    preprocess_dataset(dataset_path + '/')
    save_aligned_images(dataset_path)
    get_embeddings('./aligned_images')


def recognize_students_function(event):
    print('recognize_students_button clicked')
    global preds
    X = np.load('X.npy')
    y = np.load('y.npy')
    preprocess_image(img_path)
    preds = test(img_path, X, y)


def take_attendence_function(event):
    print('take_attendence_button clicked')
    week_number = int(app.frames[StartPage].textbox5.get())
    RecordAttendance('Attendance.xlsx', 'section 1', week_number, preds)

def MyKernel():
    global CroppedImage
    AttendanceDone=False
    CroppedImage = []
    cap = cv2.VideoCapture(0)
    MyTemplate = cap.read()
    FrameNumber = 0
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    xhTemp = 0
    yhTemp = 0
    NFrames = 9
    Counter=0
    while (True):
        ret, frame = cap.read()
        Image = frame
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = detect_image(frame, False)
        xh, yh, wh, hh = 0, 0, 0, 0
        for f in face:
            xh = f.left()
            yh = f.top()
            wh = f.right()
            hh = f.bottom()

            Image = frame
            Image = cv2.rectangle(Image, (xh, yh), (wh, hh), (255, 0, 0), 2)

            if abs(xh - xhTemp) < 20:
                Counter = Counter + 1
            else:
                Counter = 0
            if Counter == NFrames:
                # AttendanceDone=True
                ret, frame = cap.read()
                cv2.imwrite('test.jpg', frame)
                cap.release()
                cv2.destroyAllWindows()
                return frame
            xhTemp = xh
            yhTemp = yh

        cv2.imshow('frame', Image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print{FrameNumber}
            break
        FrameNumber = FrameNumber + 1

    cap.release()
    cv2.destroyAllWindows()
    # cv2.imshow('frame', CroppedImage)
    # cv2.waitKey(0)
    return None

def video(event):
    MyKernel()
    X = np.load('X.npy')
    y = np.load('y.npy')
    preds = test('test.jpg', X, y)
    print('done')
    week_number = int(app.frames[StartPage].textbox5.get())
    RecordAttendance('Attendance.xlsx', 'section 1', week_number, preds)

class SeaofBTCapp(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        container = tk.Frame(self)

        container.pack(side="top", fill="both", expand=True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (StartPage, MainPage):
            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]

        frame.tkraise()

class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Start Page", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        tk.Label(self, text="TA name :", font=LARGE_FONT).pack()
        textbox1 = tk.Entry(self)
        textbox1.pack()

        tk.Label(self, text="Year number :", font=LARGE_FONT).pack()
        textbox2 = tk.Entry(self)
        textbox2.pack()

        tk.Label(self, text="Course name :", font=LARGE_FONT).pack()
        textbox3 = tk.Entry(self)
        textbox3.pack()

        tk.Label(self, text="Section number :", font=LARGE_FONT).pack()
        textbox4 = tk.Entry(self)
        textbox4.pack()

        tk.Label(self, text="Week number :", font=LARGE_FONT).pack()
        self.textbox5 = tk.Entry(self)
        self.textbox5.pack()

        start_button = tk.Button(self, text='Start', command=lambda :controller.show_frame(MainPage))
        start_button.pack()

class MainPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        # Buttons
        browse_dataset_button = tk.Button(self, text='Browse Dataset')
        extract_embeddings_button = tk.Button(self, text='Extract Embeddings')
        camera_button = tk.Button(self, text='Take attendance by camera')
        browse_image_button = tk.Button(self, text='Browse Image')
        recognize_students_button = tk.Button(self, text='Recognize the students')
        take_attendence_button = tk.Button(self, text='take the attendance')

        # Buttons Positions
        browse_dataset_button.pack()
        extract_embeddings_button.pack()
        camera_button.pack()
        browse_image_button.pack()
        recognize_students_button.pack()
        take_attendence_button.pack()

        browse_dataset_button.bind("<Button-1>", Browse_Folder)
        extract_embeddings_button.bind("<Button-1>", extract_embeddings_function)
        camera_button.bind("<Button-1>", video)
        browse_image_button.bind("<Button-1>", browse_image)
        recognize_students_button.bind("<Button-1>", recognize_students_function)
        take_attendence_button.bind("<Button-1>", take_attendence_function)

        '''img = ImageTk.PhotoImage(Image.open("Roz.jpg"))
        panel = tk.Label(self, image=img)
        panel.pack(side="bottom", fill="both", expand="yes")'''


app = SeaofBTCapp()
app.title('Roz')
img = ImageTk.PhotoImage(Image.open("Roz.jpg"))
panel = tk.Label(app, image=img)
panel.pack(side="bottom", fill="both", expand="yes")
app.mainloop()