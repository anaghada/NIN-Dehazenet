from flask import Flask,render_template,request
import  os
import main as mn
from Eval import measures
app = Flask(__name__)

@app.route('/')
def start():
    return render_template("index.html",tottime="")


@app.route('/load')
def design():
    return render_template('index.html')


@app.route('/begin', methods=['POST'])
def begin():
    import datetime
    first_time = datetime.datetime.now()
    # filename="C:\\Users\\ANAGHA DHILEEP\\Pyc3harmProjects\\NIN_Dehazenet\\static"

    f = request.files['imgInp']
    # f.save(f.filename)
    f.save("C:\\Users\\ANAGHA DHILEEP\\PycharmProjects\\NIN_Dehazenet\\static\\image\\a.jpg")

    f2 = request.files['imgInp']
    # f.save(f.filename)
    f.save("C:\\Users\\ANAGHA DHILEEP\\PycharmProjects\\NIN_Dehazenet\\static\\image\\b.jpg")

    import cv2


    image = cv2.imread("C:\\Users\\ANAGHA DHILEEP\\PycharmProjects\\NIN_Dehazenet\\static\\image\\a.jpg", 1)
    newsize=(240,240)
    aa=cv2.resize(image,newsize)
    cv2.imwrite("C:\\Users\\ANAGHA DHILEEP\\PycharmProjects\\NIN_Dehazenet\\static\\image\\a.jpg",aa)
    cv2.imwrite("C:\\Users\\ANAGHA DHILEEP\\PycharmProjects\\NIN_Dehazenet\\static\\image\\b.jpg",aa)

    # image = cv2.imread("C:\\Users\\ANAGHA DHILEEP\\PycharmProjects\\NIN_Dehazenet\\static\\image\\b.jpg", 1)
    # newsize=(240,240)
    # bb=cv2.resize(image,newsize)
    # cv2.imwrite("C:\\Users\\ANAGHA DHILEEP\\PycharmProjects\\NIN_Dehazenet\\static\\image\\b.jpg",bb)


    mn.pdt("C:\\Users\\ANAGHA DHILEEP\\PycharmProjects\\NIN_Dehazenet\\static\\image\\a.jpg")

    h, j = measures("C:\\Users\\ANAGHA DHILEEP\\PycharmProjects\\NIN_Dehazenet\\static\\image\\a.jpg", "C:\\Users\\ANAGHA DHILEEP\\PycharmProjects\\NIN_Dehazenet\\static\\image\\c.jpg")
    # print(h,j)

    later_time = datetime.datetime.now()
    difference = later_time - first_time
    datetime.timedelta(0, 8, 562000)
    seconds_in_day = 24 * 60 * 60
    print(divmod(difference.days * seconds_in_day + difference.seconds, 60))
    res=divmod(difference.days * seconds_in_day + difference.seconds, 60)

    return render_template('index.html',pt="/static/image/c.jpg",clearimg="/static/0_clear.jpg",h=h,j=j,tottime=res)


if __name__ == '__main__':
    app.run(debug=True,port=5050)

