from nim import train, play, plot_eval
import argparse
ai, eval = train(100000)
plot_eval(eval)
#play(ai)