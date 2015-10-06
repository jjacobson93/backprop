from random import random as rand
from math import exp as e

SIGMA = 1.20301924981290
# SIGMA = 6

def sigmoid(x, sigma=SIGMA):
	return 1/(1 + e(-sigma*x))
	# try:
	# 	return (1 - e(-sigma*x))/(1 + e(-sigma*x))
	# except:
	# 	return 1 if x > 0.5 else -1

def sigmoid_prime(x, sigma=SIGMA):
	return (sigma*e(sigma*x))/((e(sigma*x) + 1)**2)
	# try:
	# 	return 2*(sigma*e(sigma*x))/((e(sigma*x) + 1)**2)
	# except:
	# 	return 0

class BackpropNet(object):
	HIDDEN_SIZE = 7
	STOPPING_VALUE = 0.1
	MAX_EPOCHS = 100000

	def __init__(self, input_size, output_size, hidden_size=HIDDEN_SIZE, V=None, W=None):
		self.input_size = input_size
		self.output_size = output_size
		self.hidden_size = hidden_size

		self.V = [[rand() for x in xrange(self.hidden_size)] for y in xrange(self.input_size)] if not V else V
		self.W = [[rand() for x in xrange(self.output_size)] for y in xrange(self.hidden_size)] if not W else W

	def train(self, samples, alpha=1, stopping_value=STOPPING_VALUE, max_epochs=MAX_EPOCHS, f=sigmoid, fprime=sigmoid_prime):
		condition = False
		epochs = 0
		while not condition:
			errors = [] # List error for all neurons for each training set
			print "Epoch", epochs #Test
			for inputs, target in samples:
				# print "\tInputs:", inputs #Test
				print "Target:", target #Test
				# Feed forward
				Z_in = [sum(inputs[i]*self.V[i][j] for i in xrange(self.input_size)) for j in xrange(self.hidden_size)]
				# print "Z_in =", Z_in
				Z = [f(j) for j in Z_in]
				Y_in = [sum(Z[j]*self.W[j][k] for j in xrange(self.hidden_size)) for k in xrange(self.output_size)]
				Y = [f(i) for i in Y_in]
				
				print "Y:", Y #Test

				# Backpropogation
				error = [target[k] - Y[k] for k in xrange(self.output_size)]
				errors.append(error)

				# delta(k) = (tk - yk)f'(yink)
				delta_Y = [(error[k])*fprime(Y_in[k]) for k in xrange(self.output_size)]
				# delta_wjk = [alpha*delta_k[j]*Z[j] for j in xrange(self.hidden_size)]
				delta_W = [[alpha*delta_Y[k]*Z[j] for j in xrange(self.hidden_size)] for k in xrange(self.output_size)]
				# delta_Z = [sum(delta_Y[k]*self.V[j][k]*fprime(Z_in[j]) for k in xrange(self.input_size)) for j in xrange(self.hidden_size)]
				delta_Z = [sum(delta_Y[k]*self.W[j][k]*fprime(Z_in[j]) for k in xrange(self.output_size)) for j in xrange(self.hidden_size)]
				delta_V = [[alpha*delta_Z[j]*inputs[i] for i in xrange(self.input_size)] for j in xrange(self.hidden_size)]

				# Adjustment of weights
				self.W = [[self.W[j][k] + alpha*delta_W[k][j] for k in xrange(self.output_size)] for j in xrange(self.hidden_size)]
				self.V = [[self.V[i][j] + alpha*delta_V[j][i] for j in xrange(self.hidden_size)] for i in xrange(self.input_size)]

			condition = True
			# print "Error Check:"
			for error in errors:
				# print "\tError:", error
				for e in error:
					if abs(e) > stopping_value:
						condition = False

			if epochs >= max_epochs:
				condition = True

			epochs += 1

		print "It took {} epochs to finish".format(epochs)

	def test(self, samples, threshold=0.2, f=sigmoid, fprime=sigmoid_prime):
		for inputs, target in samples:
			Z_in = [sum(inputs[i]*self.V[i][j] for i in xrange(self.input_size)) for j in xrange(self.hidden_size)]
			Z = [f(j) for j in Z_in]
			Y_in = [sum(Z[j]*self.W[j][k] for j in xrange(self.hidden_size)) for k in xrange(self.output_size)]
			Y = [f(i) for i in Y_in]

			# max_index = Y.index(max(Y))

			error = [abs(Y[k] - target[k]) for k in xrange(self.output_size)]
			avg_error = sum(error)/len(error)
			if avg_error < threshold:
				print "YEP! Y == target"
			else:
				print "Woops:"
			print "Y = {}".format(Y)
			print "target = {}".format(target)
