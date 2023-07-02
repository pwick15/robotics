
import random
import turtle
import os

wn = turtle.Screen()
wn.title('Pong by @punji')
wn.bgcolor("black")
wn.setup(width=800, height=600)
wn.tracer(0)  # lets us speed up the games quite a bit

# Parameters
a_score = 0
b_score = 0

# Paddle A
paddle_a = turtle.Turtle()
paddle_a.speed(0) # this is the speed of animation, sets the speed to max speed
paddle_a.shape('square')
paddle_a.color('pink')
paddle_a.penup()  # we dont need to draw a line
paddle_a.goto(-350, 0)
paddle_a.shapesize(stretch_wid=5, stretch_len=1)

# Paddle B
paddle_b = turtle.Turtle()
paddle_b.speed(0)
paddle_b.shape('square')
paddle_b.color('blue')
paddle_b.penup()  # we dont need to draw a line
paddle_b.goto(350, 0)
paddle_b.shapesize(stretch_wid=5, stretch_len=1)

# Ball
ball = turtle.Turtle()
ball.speed(0)  # this is the speed of animation, sets the speed to max speed
ball.shape('square')
ball.color('white')
ball.penup()  # we dont need to draw a line
ball.goto(0, 0)
ball.dx = 4
ball.dy = 4

# Pen
pen = turtle.Turtle()
pen.speed(0)
pen.color("white")
pen.penup()
pen.hideturtle()
pen.goto(0,260)
pen.write(f'Player A: {a_score}  Player B: {b_score}', align="center", font=("Courier", 24, "normal"))



# Function
def paddle_a_up():
    # need to know current y coordinate
    y = paddle_a.ycor()  # returns the y-coordinate
    y += 100
    paddle_a.sety(y)


def paddle_a_down():
    # need to know current y coordinate
    y = paddle_a.ycor()  # returns the y-coordinate
    y -= 100
    paddle_a.sety(y)


def paddle_b_up():
    # need to know current y coordinate
    y = paddle_b.ycor()  # returns the y-coordinate
    y += 100
    paddle_b.sety(y)


def paddle_b_down():
    # need to know current y coordinate
    y = paddle_b.ycor()  # returns the y-coordinate
    y -= 100
    paddle_b.sety(y)


# keyboard binding
wn.listen()
wn.onkeypress(paddle_a_up, "w")
wn.onkeypress(paddle_a_down, "s")
wn.onkeypress(paddle_b_up, "Up")
wn.onkeypress(paddle_b_down, "Down")


# every game needs a main game loop which is where the main code goes
while True:
    wn.update()  # every time the loop runs, it updates the screen

    # move the ball
    ball.setx(ball.xcor() + ball.dx)
    ball.sety(ball.ycor() + ball.dy)

    # border checking (once it gets to a certain point, we want the ball to bounce)
    # the height is 600 -> 300 above the middle. ball height is 20, therefore when the ball's height > 300-20, we want it to bounce
    if ball.ycor() > 290:
        ball.dy *= -1
        os.system("afplay bounce.wav&")

    if ball.ycor() < -280:
        ball.dy *= -1
        os.system("afplay bounce.wav&")



    # if the ball leaves the screen, reset it
    if ball.xcor() > 390:
        ball.goto(0,random.randint(-200,200))
        ball.dx*=-1
        a_score += 1
        pen.clear()
        pen.write(f'Player A: {a_score}  Player B: {b_score}', align="center", font=("Courier", 24, "normal"))


    if ball.xcor() < -390:
        ball.goto(0,random.randint(-200,200))
        ball.dx*=-1
        b_score += 1
        pen.clear()
        pen.write(f'Player A: {a_score}  Player B: {b_score}', align="center", font=("Courier", 24, "normal"))


    # Paddle and ball collisions
    if ball.xcor() > 330 and ball.xcor() < 340 and ball.ycor() < paddle_b.ycor() + 50 and ball.ycor() > paddle_b.ycor() - 50:
        ball.dx *= -1
        os.system("afplay bounce.wav&")


    if ball.xcor() < -330 and ball.xcor() > -340 and ball.ycor() < paddle_a.ycor() + 50 and ball.ycor() > paddle_a.ycor() - 50:
        ball.dx *= -1
        os.system("afplay bounce.wav&")

    
    # Ensure the two paddles stay within the screen
    if paddle_a.ycor() > 250:
        paddle_a.sety(250)
    
    if paddle_a.ycor() < -250:
        paddle_a.sety(-250)
    
    if paddle_b.ycor() > 250:
        paddle_b.sety(250)

    if paddle_b.ycor() < -250:
        paddle_b.sety(-250)

    