from PIL import Image, ImageTk
from tkinter import Label, Tk, Entry, Text
from tkinter import Button
from tkinter import messagebox
import pydicom
import numpy as np
from tkinter import filedialog
import os
from pydicom import Dataset, DataElement
import tkinter
from ct_model_class import Model
import cv2
import os
from PIL import Image
from torchvision.transforms.v2 import Resize
import torch


scan_dir = []
scan_number = 0
file_number = 0
total_file_num = 0
current_file_num = 1
data_type = ''
ct_scan_arr = []
entered_ID = ''
metadata = ['(0008, 0020)',
            '(0008, 0030)',
            '(0008, 0070)',
            '(0008, 103e)',
            '(0010, 0010)',
            '(0010, 0030)',
            '(0010, 0040)',
            '(0010, 1010)',
            '(0018, 0015)',
            '(0018, 0090)',
            '(0018, 5100)',
            '(0018, 9311)']
model = Model()
transform = Resize((512, 512))

def read_dcm(path):
    global data_type
    print(path)
    data = pydicom.dcmread(path)
    arr = data.pixel_array
    if len(arr.shape) == 2:
        data_type = 'single'
        return arr
    elif len(arr.shape) == 3:
        data_type = 'multiple'
        return arr

def read_dcm_len(path):
    global data_type
    data = pydicom.dcmread(path)
    arr = data.pixel_array
    if len(arr.shape) == 2:
        return 1
    elif len(arr.shape) == 3:
        return len(arr)
    
def apply_model(arr):
    global model, transform
    
    img = np.array(arr, dtype = float) 
    img = (img - img.min()) / (img.max() - img.min()) * 255.0  
    img = img.astype(np.uint8)
    img = np.expand_dims(img, 2)
    img = np.dstack((img, img[:, :, 0], img[:, :, 0]))

    aorta, arterial = model.process_images(img)
    
    PIL_scan = Image.fromarray(img)
    PIL_scan.putalpha(255)
    
    if aorta[1].item() > 9.5:
        aorta_mask = np.expand_dims(aorta[0], 2)
        aorta_mask = np.dstack((aorta_mask, aorta_mask[:, :, 0], aorta_mask[:, :, 0]))
        aorta_mask[:, :, 1] = 0
        aorta_mask[:, :, 2] = 0
        aorta_mask = np.array(transform(torch.Tensor(aorta_mask).transpose(2, 1).transpose(1, 0)).transpose(1, 0).transpose(1, 2), dtype=np.uint8)
        
        PIL_aorta = Image.fromarray(aorta_mask)
        PIL_aorta.putalpha(50)
        PIL_scan = Image.alpha_composite(PIL_scan, PIL_aorta)
        
        
    if arterial[1].item() > 9.5:
        arterial_mask = np.expand_dims(arterial[0], 2)
        arterial_mask = np.dstack((arterial_mask, arterial_mask[:, :, 0], arterial_mask[:, :, 0]))
        arterial_mask[:, :, 0] = 0
        arterial_mask[:, :, 1] = 0
        arterial_mask = np.array(transform(torch.Tensor(arterial_mask).transpose(2, 1).transpose(1, 0)).transpose(1, 0).transpose(1, 2), dtype=np.uint8)
        
        PIL_arterial = Image.fromarray(arterial_mask)
        PIL_arterial.putalpha(50)
        PIL_scan = Image.alpha_composite(PIL_scan, PIL_arterial)
    
    PIL_scan = PIL_scan.convert("RGB")
    
    return np.uint8(PIL_scan)

        
def upload_file(main_window):
    global scan_dir, file_number, scan_number, ct_scan_arr, data_type, entered_ID, metadata
    if entered_ID != '':
        path = filedialog.askdirectory()
        file_number = 0
        scan_number = 0
        scan_dir = sorted(os.listdir(path))
        for file_name in scan_dir:
            if 'dcm' not in str(file_name):
                scan_dir.remove(file_name)
        for i in range(len(scan_dir)):
            scan_dir[i] = path + '/' + scan_dir[i]
        os.mkdir('Облако' + '/ID_' + entered_ID)
        for i in range(len(scan_dir)):
            cur_data = pydicom.dcmread(scan_dir[i])
            cur_data[('0010', '0010')] = DataElement(0x00100010, 'PN', entered_ID)
            cur_data[('0008', '1010')] = DataElement(0x00081010, 'SH', '')
            cur_data[('0008', '1010')] = DataElement(0x00081010, 'LO', '')
            cur_data.save_as('Облако' + '/ID_' + entered_ID + '/' + scan_dir[i].split('/')[-1])
            scan_dir[i] = 'Облако' + '/ID_' + entered_ID + '/' + scan_dir[i].split('/')[-1]
        data = pydicom.dcmread(scan_dir[0])
        report_text = ''
        for line in data:
            data_line = ' '.join(str(line).split()) + '\n'
            if data_line[0:12] in metadata:
                report_text += data_line

        Report = Label(main_window, bg='#D9D9D9', text=report_text, font=('Microsoft YaHei UI Light', 14), anchor='nw', justify='left')
        Report.place(x=1000, y=120, width=400, height=515)
        entered_ID = ''
        ct_scan_arr = read_dcm(scan_dir[file_number])

        refresh_img(main_window)

def open_file(main_window, path=''):
    global scan_dir, file_number, scan_number, ct_scan_arr, data_type

    path = filedialog.askdirectory()
    file_number = 0
    scan_number = 0
    scan_dir = sorted(os.listdir(path))
    for file_name in scan_dir:
        if 'dcm' not in str(file_name):
            scan_dir.remove(file_name)

    data = pydicom.dcmread(path + '/' + scan_dir[0])
    report_text = ''
    for line in data:
        data_line = ' '.join(str(line).split()) + '\n'
        if data_line[0:12] in metadata:
            report_text += data_line
    Report = Label(main_window, bg='#D9D9D9', text=report_text, font=('Microsoft YaHei UI Light', 14), anchor='nw', justify='left')
    Report.place(x=1000, y=120, width=400, height=515)
    for i in range(len(scan_dir)):
        scan_dir[i] = path + '/' + scan_dir[i]
    ct_scan_arr = read_dcm(scan_dir[file_number])

    refresh_img(main_window)


def show_next_img(main_window):
    global scan_dir, file_number, scan_number, ct_scan_arr

    # if scan_dir != []:
    if data_type == 'multiple':
        if scan_number < len(read_dcm(scan_dir[file_number])) - 1:
            scan_number += 1
        if scan_number == len(read_dcm(scan_dir[file_number])) - 1 and file_number < len(scan_dir) - 1:
            file_number += 1
            ct_scan_arr = read_dcm(scan_dir[file_number])
    else:
        if file_number < len(scan_dir) - 1:
            file_number += 1
            ct_scan_arr = read_dcm(scan_dir[file_number])
    refresh_img(main_window)

def show_prev_img(main_window):
    global scan_dir, file_number, scan_number, ct_scan_arr

    if data_type == 'multiple':
        if scan_number > 0:
            scan_number -= 1
            refresh_img(main_window)
        if scan_number == 0 and file_number > 0:
            file_number -= 1
            ct_scan_arr = read_dcm(scan_dir[file_number])
    else:
        if file_number > 0:
            file_number -= 1
            ct_scan_arr = read_dcm(scan_dir[file_number])
    refresh_img(main_window)

def refresh_img(main_window):
    global scan_dir, file_number, scan_number, ct_scan_arr, data_type
    # ct_scan = ImageTk.PhotoImage(Image.open(scan_dir[scan_number]))
    if data_type == 'multiple':
        ct_scan = ImageTk.PhotoImage(Image.fromarray(apply_model(ct_scan_arr[scan_number])))
    else:
        ct_scan = ImageTk.PhotoImage(Image.fromarray(apply_model(ct_scan_arr)))
    panel = Label(main_window, image=ct_scan)
    panel.place(x=main_window.winfo_screenwidth() // 2 - ct_scan.width() // 2, y=120)

    scan_counter = Label(main_window, text=str(current_file_num) + ' \\ ' + str(total_file_num), font=('Microsoft YaHei UI Light', 14))
    scan_counter.place(x=200, y=340, width=150)
    main_window.mainloop()

def scroll_to_num(main_window, demanded_scan_num):
    global scan_dir, ct_scan_arr, scan_number, file_number, current_file_num

    scan_number = 0
    file_number = 0
    current_file_num = demanded_scan_num
    while demanded_scan_num > read_dcm_len(scan_dir[file_number]):
        demanded_scan_num -= read_dcm_len(scan_dir[file_number])
        file_number += 1
    scan_number = demanded_scan_num - 1
    ct_scan_arr = read_dcm(scan_dir[file_number])
    refresh_img(main_window)

def launch_main_window():
    global scan_dir, scan_number, file_number

    main_window = Tk()
    width = main_window.winfo_screenwidth()
    height = main_window.winfo_screenheight()
    main_window.title('App')
    main_window.geometry("%dx%d" % (width, height))
    main_window.config(bg='white')

    # Labels
    instuments_label = Label(main_window, text='Инструменты', font=('Microsoft YaHei UI Light', 14))
    scan_label = Label(main_window, text='Сканы', font=('Microsoft YaHei UI Light', 14))
    report_label = Label(main_window, text='Метаданные', font=('Microsoft YaHei UI Light', 14))
    instuments_label.place(x=200, y=100)
    scan_label.place(x=695, y=100)
    report_label.place(x=1150, y=100)

    # Buttons
    open_file_button = Button(text='Загрузить', command= lambda: open_file(main_window))
    open_file_button.place(x=200, y=130, width=150)

    next_img_button = Button(text='Следующий', command= lambda: show_next_img(main_window))
    next_img_button.place(x=200, y=160, width=150)

    prev_img_button = Button(text='Предыдущий', command=lambda: show_prev_img(main_window))
    prev_img_button.place(x=200, y=190, width=150)

    # Text Widgets
    Report = Text(main_window, bg='#D9D9D9')
    Report.place(x=1000, y=120, width=400, height=100)

    def on_enter_ID(e):
        ID_entry.delete(0, 'end')

    def on_leave_ID(e):
        name = ID_entry.get()
        if name == '':
            ID_entry.insert(0, 'ID')

    def on_enter_scan(e):
        Scan_num_entry.delete(0, 'end')

    def on_leave_scan(e):
        name = ID_entry.get()
        if name == '':
            Scan_num_entry.insert(0, 'ID')

    def enter_ID():
        global entered_ID
        entered_ID = ID_entry.get()
        ID_entry.delete(0, 'end')
        upload_file(main_window)

    def enter_scan_num():
        demanded_scan_num = int(Scan_num_entry.get())
        Scan_num_entry.delete(0, 'end')
        scroll_to_num(main_window, demanded_scan_num)

    ID_entry = Entry(main_window, border=0, fg='black', bg='#D9D9D9', font=('Microsoft YaHei UI Light', 11))
    ID_entry.place(x=200, y=280, width=150)
    ID_entry.insert(0, 'ID')
    ID_entry.bind('<FocusIn>', on_enter_ID)
    ID_entry.bind('<FocusOut>', on_leave_ID)

    enter_ID_button = Button(main_window, width=200, text='Выгрузить', bg='#53A2FF', command=enter_ID)
    enter_ID_button.place(x=200, y=310, width=150)

    Scan_num_entry = Entry(main_window, border=0, fg='black', bg='#D9D9D9', font=('Microsoft YaHei UI Light', 11))
    Scan_num_entry.place(x=200, y=220, width=150)
    Scan_num_entry.insert(0, 'Номер скана')
    Scan_num_entry.bind('<FocusIn>', on_enter_scan)
    Scan_num_entry.bind('<FocusOut>', on_leave_scan)

    scan_jump = Button(main_window, width=200, text='Перейти', bg='#53A2FF', command=enter_scan_num)
    scan_jump.place(x=200, y=250, width=150)

    main_window.mainloop()