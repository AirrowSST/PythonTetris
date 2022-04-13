# PythonTetris
An implementation of Tetris in Python 3.10

(Note: numpy, as well as the default python libraries, will be needed to use the code)

This file is not an executable file. It contains classes that implement Tetris in python. The implementation is based on Tetr.io's implementation of Tetris, but the code can be modified to change the details of the implementation. The code is well-documented inside the python file. The code includes functionality for console printout. an example of how you would use the code is given below (in this example, threading is used to simulate inputs into the game):
    
    from Tetris import *
    def test_console_graphics():
        game = Game(autonomous=True)
        game.score.set_level(5)
        controller = Controller(game)
        graphics = ConsoleGraphics(game)  # automatically prints output to the console
        
        # simulates inputs
        threading.Timer(0.1, controller.right).start()
        threading.Timer(0.2, controller.right).start()
        threading.Timer(0.3, controller.right).start()
        threading.Timer(0.4, controller.right).start()
        threading.Timer(0.5, controller.right).start()
        threading.Timer(1, controller.hard_drop).start()
        threading.Timer(1.5, controller.hold).start()
        threading.Timer(1.9, controller.left).start()
        game.start()
        while not game.is_dead:
            pass
