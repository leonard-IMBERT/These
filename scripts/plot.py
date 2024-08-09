off_book=False

try:
  get_ipython()
  off_book=False
except NameError:
  off_book=True

print(off_book)

#|%%--%%| <u0TUQhpnj1|fdmd5ofewv>
r"""°°°
# Common imports
°°°"""
#|%%--%%| <fdmd5ofewv|EP8vVkRALZ>

from typing import Callable, TypeVar, List
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from os import path


#|%%--%%| <EP8vVkRALZ|AeZn7ko7i0>
r"""°°°
# Loss plots
°°°"""
#|%%--%%| <AeZn7ko7i0|x6FVXbbPpt>

T = TypeVar('T', float, npt.NDArray[np.float64])
def mse(x: T, y: float) -> T:
  return (x - y)**2

def d_mse(x: T, y: float) -> T:
  return 2 * (x - y)

def mae(x: T, y: float) -> T:
  if isinstance(x, float):
    return abs(x - y)
  return np.abs(x - y)

def d_mae(x: T, y: float) -> T:
  if isinstance(x, float):
    if (x - y) > 0:
      return 1
    if (x - y) < 0:
      return -1
    return 0
  return (x - y) / (np.abs(x - y) + 1e-7)

def explo(x: T, y: float) -> T:
  return ((x-y) ** 2)/2 + np.exp(4 * (x-y))

def d_explo(x: T, y: float) -> T:
  return (x-y) + 4 * np.exp(4 * (x-y))


class SGD:
  """Emulation of a simple SGD optimiser on a single parameter theta"""
  def __init__(self, lr: float, initial_value: float, loss_f: Callable[[float], float], d_loss_f: Callable[[float], float]):
    self._lr = lr
    self._theta_values: List[float] = [initial_value]
    self._loss_f = loss_f
    self._d_loss_f = d_loss_f
    self._loss_values: List[float] = [self._loss_f(self._theta_values[-1])]
    self._grad_values: List[float] = [self._d_loss_f(self._theta_values[-1])]

  def step(self) -> None:
    """Take a step in the optimisation process"""
    self._theta_values.append(self._theta_values[-1] - self._grad_values[-1] * self._lr)
    self._loss_values.append(self._loss_f(self._theta_values[-1]))
    self._grad_values.append(self._d_loss_f(self._theta_values[-1]))

  def get_theta(self) -> npt.NDArray[np.float64]:
    """Return the parameter values during the optimisation"""
    return np.array(self._theta_values, dtype=np.float64)

  def get_l(self) -> npt.NDArray[np.float64]:
    """Return the loss values during the optimisation"""
    return np.array(self._loss_values, dtype=np.float64)

  def get_g(self) -> npt.NDArray[np.float64]:
    """Return the gradient values during the optimisation"""
    return np.array(self._grad_values, dtype=np.float64)

  def get_lr(self) -> float:
    """Return the learning rate of the SGD"""
    return self._lr





#|%%--%%| <x6FVXbbPpt|eYAwUQ6g65>
r"""°°°
## MSE loss function
°°°"""
#|%%--%%| <eYAwUQ6g65|45swQoLX5v>



TRAIN_STEP = 20

TARGET = 2

def mse_t(x: float):
  return mse(x, TARGET)

def d_mse_t(x: float):
  return d_mse(x, TARGET)


big_sgd = SGD(lr=1 - 5e-2, initial_value=0.25, loss_f=mse_t, d_loss_f=d_mse_t)
small_sgd = SGD(lr=1e-1, initial_value=0.25, loss_f=mse_t, d_loss_f=d_mse_t)
too_small_sgd = SGD(lr=4e-2, initial_value=3.75, loss_f=mse_t, d_loss_f=d_mse_t)





for ii in range(0, TRAIN_STEP):
  big_sgd.step()
  small_sgd.step()
  too_small_sgd.step()


considered_range = np.arange(TARGET - 2, TARGET + 2 + 0.01, 0.01)

if off_book:
  fig, (ax_1, ax_2) = plt.subplots(2, 1, sharex=True, height_ratios=(3/4, 1/4), gridspec_kw={'hspace': 0}, figsize=(8,6), dpi=300)
else:
  fig, (ax_1, ax_2) = plt.subplots(2, 1, sharex=True, height_ratios=(3/4, 1/4), gridspec_kw={'hspace': 0})

ax_1.plot(considered_range, mse(considered_range, TARGET), label="Loss function")
ax_1.plot(big_sgd.get_theta(), big_sgd.get_l(), '.--', color='tab:green',
         label=f"$\\theta$ values at each step (LR = {big_sgd.get_lr():.2f})")
ax_1.plot(small_sgd.get_theta(), small_sgd.get_l(), '.--', color='tab:orange',
         label=f"$\\theta$ values at each step (LR = {small_sgd.get_lr():.2f})")
ax_1.plot(too_small_sgd.get_theta(), too_small_sgd.get_l(), '.--', color='tab:red',
         label=f"$\\theta$ values at each step (LR = {too_small_sgd.get_lr():.2f})")
ax_1.legend()
ax_1.set_title("Optimisation example example of one parameter\n$\\theta$ using SGD for different learning rate (LR)")
ax_2.plot(considered_range, d_mse(considered_range, TARGET))

for ii in range(6):
    ax_1.text(big_sgd.get_theta()[ii], big_sgd.get_l()[ii], f'Step {ii}',
             color='tab:green', horizontalalignment="left", verticalalignment="bottom")
    ax_1.text(small_sgd.get_theta()[ii], small_sgd.get_l()[ii], f'Step {ii}',
             color='tab:orange',  horizontalalignment="right", verticalalignment="top")
    ax_1.text(too_small_sgd.get_theta()[ii], too_small_sgd.get_l()[ii], f'Step {ii}',
             color='tab:red',  horizontalalignment="left", verticalalignment="top")
ax_1.grid(True)
ax_2.grid(True)
ax_2.set_xlabel(r"$\theta$")
ax_1.set_ylabel(r"$\mathcal{L}$")
ax_2.set_ylabel(r"$\frac{\partial \mathcal{L}}{\partial \theta}$")
if off_book:
  fig.savefig(path.join("plots", "MSE_illustration.png"))

#|%%--%%| <45swQoLX5v|d0v0eQxbWS>
r"""°°°
## MAE loss function
°°°"""
#|%%--%%| <d0v0eQxbWS|wJhz4ByIzG>




TRAIN_STEP = 10

TARGET = 2

def mae_t(x: float):
    return mae(x, TARGET)

def d_mae_t(x: float):
    return d_mae(x, TARGET)


big_sgd = SGD(lr=1 - 3e-2, initial_value=0.25, loss_f=mae_t, d_loss_f=d_mae_t)
small_sgd = SGD(lr=0.8, initial_value=0.25, loss_f=mae_t, d_loss_f=d_mae_t)
too_small_sgd = SGD(lr=5e-2, initial_value=3.25, loss_f=mae_t, d_loss_f=d_mae_t)




for ii in range(0, TRAIN_STEP):
  big_sgd.step()
  small_sgd.step()
  too_small_sgd.step()


considered_range = np.arange(TARGET - 2, TARGET + 2 + 0.01, 0.01)

if off_book:
  fig, (ax_1, ax_2) = plt.subplots(2, 1, sharex=True, height_ratios=(3/4, 1/4), gridspec_kw={'hspace': 0}, figsize=(8,6), dpi=300)
else:
  fig, (ax_1, ax_2) = plt.subplots(2, 1, sharex=True, height_ratios=(3/4, 1/4), gridspec_kw={'hspace': 0})
ax_1.plot(considered_range, mae(considered_range, TARGET), label="Loss function")
ax_1.plot(big_sgd.get_theta(), big_sgd.get_l(), '.--', color='tab:green',
         label=f"$\\theta$ values at each step (LR = {big_sgd.get_lr():.2f})")
ax_1.plot(small_sgd.get_theta(), small_sgd.get_l(), '.--', color='tab:orange',
         label=f"$\\theta$ values at each step (LR = {small_sgd.get_lr():.2f})")
ax_1.plot(too_small_sgd.get_theta(), too_small_sgd.get_l(), '.--', color='tab:red',
         label=f"$\\theta$ values at each step (LR = {too_small_sgd.get_lr():.2f})")
ax_1.legend()
ax_1.set_title("Optimisation example example of one parameter\n$\\theta$ using SGD for different learning rate (LR)")
ax_2.plot(considered_range, d_mae(considered_range, TARGET))

for ii in range(6):
  ax_1.text(big_sgd.get_theta()[ii], big_sgd.get_l()[ii], f'Step {ii}',
           color='tab:green', horizontalalignment="left", verticalalignment="bottom")
  ax_1.text(small_sgd.get_theta()[ii], small_sgd.get_l()[ii], f'Step {ii}',
           color='tab:orange',  horizontalalignment="right", verticalalignment="top")
  ax_1.text(too_small_sgd.get_theta()[ii], too_small_sgd.get_l()[ii], f'Step {ii}',
           color='tab:red',  horizontalalignment="left", verticalalignment="top")
ax_1.grid(True)
ax_2.grid(True)
ax_2.set_xlabel(r"$\theta$")
ax_1.set_ylabel(r"$\mathcal{L}$")
ax_2.set_ylabel(r"$\frac{\partial \mathcal{L}}{\partial \theta}$")
if off_book:
  fig.savefig(path.join("plots", "MAE_illustration.png"))


#|%%--%%| <wJhz4ByIzG|HE3J4bflNW>
r"""°°°
# Loss explosion
°°°"""
#|%%--%%| <HE3J4bflNW|VKRxvrNL0N>

TARGET = 0
TRAIN_STEP = 5

def explo_t(x: float):
  return explo(x, TARGET)

def d_explo_t(x: float):
  return d_explo(x, TARGET)

explosive_sgd = SGD(lr=1.05, initial_value=-4, loss_f=explo_t, d_loss_f=d_explo_t)

for ii in range(TRAIN_STEP):
  explosive_sgd.step()

considered_range = np.arange(TARGET - 5, TARGET + 1 + 0.01, 0.01)

if off_book:
  fig, (ax_1, ax_2) = plt.subplots(2, 1, sharex=True, height_ratios=(3/4, 1/4), gridspec_kw={'hspace': 0}, figsize=(8,6), dpi=300)
else:
  fig, (ax_1, ax_2) = plt.subplots(2, 1, sharex=True, height_ratios=(3/4, 1/4), gridspec_kw={'hspace': 0})
ax_1.set_ylim([-0.1,15])
ax_1.set_xlim([TARGET - 5, TARGET + 1.1])
ax_1.plot(considered_range, explo(considered_range, TARGET), label="Loss function")
ax_1.plot(explosive_sgd.get_theta(), explosive_sgd.get_l(), '.--', color='tab:green',
         label=f"$\\theta$ values at each step (LR = {explosive_sgd.get_lr():.2f})")

th = explosive_sgd.get_theta()
ls = explosive_sgd.get_l()
ax_1.legend()
ax_1.set_title("Optimisation example example of one parameter\n$\\theta$ using SGD")
ax_2.plot(considered_range, d_explo(considered_range, TARGET))
ax_2.set_ylim([-2.1, 3.6])

for ii in range(4):
  if ii == 2:
    continue
  ax_1.text(explosive_sgd.get_theta()[ii], explosive_sgd.get_l()[ii], f'Step {ii}',
           color='tab:green', horizontalalignment="left", verticalalignment="top")
ax_1.grid(True)
ax_2.grid(True)
ax_2.set_xlabel(r"$\theta$")
ax_1.set_ylabel(r"$\mathcal{L}$")
ax_2.set_ylabel(r"$\frac{\partial \mathcal{L}}{\partial \theta}$")
if off_book:
  fig.savefig(path.join("plots", "MSE_explosion_illustration.png"))
