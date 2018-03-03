class SDG:
	def __init__(self, lr=0.01):
		self.lr = lr # 학습률을 의미한다.

	def update(self, param, grads):
		for key in params.keys():
			params[key] -= self.lr * grads[key] # 가중치 매개변수 -= 학습률 * 기울기

class Momentum:
	def __init__(self, lr=0.01, momentum=0.9):
		self.lr = lr
		self.momentum = momentum
		self.v = None # 물체의 속도, 초기화 때는 아무 값도 담지 않음

	def update(self, params, grads):
		if self.v is None:
			self.v = {}
			for key, val in paras.items():
				self.v[key] =np.zeros_like(val) # update 시에 None 이면 매개변수와 같은 구조의 데이터로, 0을 채워서 생성

		for key in params.keys():
			# self.v[key]는 이전 update때 움직였던 가중치값(항상 마이너스)
			self.v[key] = self.momentum * self.v[key] - self.lr*grads[key]
			params[key] += self.v[key]

class AdaGrad:
	def __init__(self, lr=0.01):
		self.lr = lr
		self.h = None

	def update(self, paras, grads):
		if self.h is None
			self.h = {}
			for key, val in params.items():
				self.h[key] = np.zeros_like(val)

		for key in params.keys():
			self.h[key] += grads[key] * grads[key]
			paras[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
