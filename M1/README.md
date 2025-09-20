# Проект М1. Полет камня

## Условие
Камень бросают под углом к горизонту в земном поле
тяжести с некоторой начальной скоростью. Составьте программу, которая
численно решала бы дифференциальное уравнение движения камня,
рассчитывая его траекторию движения и определяла бы точку падения.
Исследуйте, как изменяется характер траектории при различных начальных
параметрах броска (угол, начальная скорость, коэффициент сопротивления).
Для силы сопротивления воздуха рассмотрите две модели: а) вязкое трение,
пропорциональное скорости 𝐹 ∼ 𝑣 , б) лобовое сопротивление,
пропорциональное квадрату скорости

## Решение
Для движения камня применяем 2 закон Ньютона:
$\vec{F} = m \vec{a}$

Запишем закон Ньютона в векторном виде и запишем проекции на оси X и Y. 

$$
\begin{cases}
m\vec{a_x} = F_{сопр x} \\
m\vec{a_y} = F_g + F_{сопр y}
\end{cases}
$$

1)Рассмотрим случай A, сила сопротивления воздуха пропорциональна скорости. 
Запишем закон Ньютона

$$
\begin{cases}
m\vec{a_x} = 0 - k\vec{u_x} \\
m\vec{a_y} = -mg - k\vec{u_y}
\end{cases}
$$

,где $k\vec{u_x}$ - сила вязкого сопротивления по оси x, $k\vec{u_y}$ - сила
вязкого сопротивления по оси y, $k$ - коэфицент вязкого сопротивления.

$$
\begin{cases}
m \frac{d\vec{u_x}}{dt} = -k\vec{u_x} \\
m \frac{d\vec{u_y}}{dt} = -mg - k\vec{u_y}
\end{cases} \;\Rightarrow\;
\begin{cases}
\frac{d\vec{u_x}}{dt} = -\frac{k}{m} u_x \\
\frac{d\vec{u_y}}{dt} = -g - \frac{k}{m} u_y
\end{cases}
$$ ,пусть p = $\frac{k}{m}$ 

$$
\begin{cases}
\frac{d\vec{u_x}}{dt} = -p u_x \\
\frac{d\vec{u_y}}{dt} = -g - p u_y
\end{cases} \;\Rightarrow\;
\begin{cases}
\frac{d\vec{u_x}}{\vec{u_x}} = -pdt \\
\frac{d\vec{u_y}}{-g - p \vec{u_y}} = dt
\end{cases}
$$ Проинтегрируем и выразим $\vec{u_x}$ и $\vec{u_y}$
$$
\begin{cases}
\int \frac{d\vec{u_x}}{\vec{u_x}} = \int -pdt \\
\int \frac{d\vec{u_y}}{-g - p \vec{u_y}} = \int dt
\end{cases} \;\Rightarrow\;
\begin{cases}
\int \frac{d\vec{u_x}}{\vec{u_x}} = -\int pdt \\
\int \frac{d\vec{u_y}}{g + p \vec{u_y}} = -\int dt
\end{cases}\;\Rightarrow\;
\begin{cases}
ln{|u_x|} = -pt + C\\
-\frac{1}{p}ln{|g + pu_y|} = -t + C
\end{cases}
$$

Предположим, что $u_x(0) = u_{0x}$ и $u_y(0) = u_{0y}$. Выразим $u_x(t)$ и $u_y(t)$:
$$
\begin{cases}
u_x(t) = e^{-pt} u_{0x}\\
u_y(t) = \frac{(pu_{0y}+g)e^{-pt} - g}{p}
\end{cases}
$$

Выразим координаты
$$
\begin{cases}
\int \frac{dx}{dt} \, dt = \int e^{-pt} u_{0x} dt\\
\int \frac{dy}{dt} \, dt = \int \frac{(pu_{0y}+g)e^{-pt} - g}{p}dt
\end{cases}
$$

$$
\begin{cases}
x(t) = - \frac{u_{0x}}{p}e^{-pt} + C\\
y(t) = (\frac{pu_{0y}+g}{p})(-\frac{e^{-pt}}{p})-\frac{g}{p}t + C
\end{cases}
$$

Предполагаем, что $x(0) = 0$ и $y(0) = 0$ и выражаем C:

$$
\begin{cases}
x(t) = - \frac{u_{0x}}{p}e^{-pt} + \frac{u_{0x}}{p} + x_0\\
y(t) = (\frac{pu_{0y}+g}{p^2})(-{e^{-pt}}+1)-\frac{g}{p}t + y_0
\end{cases}
$$

$p = \frac{k}{m},u_{0x} = u_0 \cos\alpha, u_{0y} = u_0 \sin\alpha, \alpha$ - угол между броском и осью X.

$$
\begin{cases}
x(t) = - \frac{(u_0 \cos\alpha)m}{k}e^{-\frac{kt}{m}} + \frac{(u_0 \cos\alpha)m}{k} + x_0\\
y(t) = (\frac{k(u_0 \sin\alpha)+gm^2}{k^2})(-e^{-\frac{kt}{m}}+1)-\frac{gm}{k}t + y_0
\end{cases}
$$

2)Рассмотрим случай, где сопротивление пропорционально квадрату скорости $F \sim u^2$
Сила сопротивления $F_{сопр} = k \vec{u} |u|$, где $|u| = \sqrt{u_x^2 + u_y^2}$

$$
\begin{cases}
m a_x = - k u_x \sqrt{u_x^2 + u_y^2} \\
m a_y = -m g - k u_y \sqrt{u_x^2 + u_y^2}
\end{cases} \;\Rightarrow\;
\begin{cases}
m \frac{du_x}{dt} = - k u_x \sqrt{u_x^2 + u_y^2} \\
m \frac{du_y}{dt} = -m g - k u_y \sqrt{u_x^2 + u_y^2}
\end{cases}\
$$

Полученная система является системой нелинейных дифференциальных уравнений, так как сила сопротивления зависит 
от квадрата скорости. В общем виде такую системы решить мы не можем, поэтому для решения этой системы
воспользуемся численными методами.

