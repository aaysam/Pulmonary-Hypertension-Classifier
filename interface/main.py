from tkinter import *
from tkmacosx import Button
from tkinter import messagebox
import main_window


login_window = Tk()
login_window.title('Регистрация')
width = login_window.winfo_screenwidth()
height = login_window.winfo_screenheight()
login_window.geometry("%dx%d" % (width, height))
login_window.configure(bg='#fff')

human_img = PhotoImage(file='human_icon.png')
Label(login_window, image=human_img, bg='white').place(x=width // 2 - human_img.width() // 2, y=100)

def signin():
    username = user.get()
    code = password.get()
    if username == 'admin' and code == '1234' or True:
        login_window.destroy()
        main_window.launch_main_window()
    else:
        messagebox.showerror('Invalid', 'Invalid username or password')

# user entry
def on_enter(e):
    user.delete(0, 'end')

def on_leave(e):
    name = user.get()
    if name == '':
        user.insert(0, 'Логин')

user = Entry(login_window, width=50, border=0, fg='black', bg='#D9D9D9', font=('Microsoft YaHei UI Light', 11))
user.place(x=width // 2 - 100, y=100 + human_img.height() + 20, width=200)
user.insert(0, 'Логин')
user.bind('<FocusIn>', on_enter)
user.bind('<FocusOut>', on_leave)

# password entry
def on_enter(e):
    password.delete(0, 'end')

def on_leave(e):
    name = password.get()
    if name == '':
        password.insert(0, 'Пароль')

password = Entry(login_window, width=41, border=0, fg='black', bg='#D9D9D9', font=('Microsoft YaHei UI Light', 11))
password.place(x=width // 2 - 100, y=100 + human_img.height() + 50, width=200)
password.insert(0, 'Пароль')
password.bind('<FocusIn>', on_enter)
password.bind('<FocusOut>', on_leave)

# sign in button
sign_in_button = Button(login_window, width=20, text='Войти', bg='#53A2FF', command=signin)
sign_in_button.place(x=width // 2 - 100, y=100 + human_img.height() + 80, width=200)

login_window.mainloop()
