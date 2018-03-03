class SDG:
	def __init__(self, lr=0.01):
		self.lr = lr # 학습률을 의미한다.

	def update(self, param, grads):
		for key in params.keys():
			params[key] -= self.lr * grads[key] # 가중치 매개변수 -= 학습률 * 기울기
