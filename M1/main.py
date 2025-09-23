import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def stone_motion(state, time, mass, resistance_coef, angle_deg, initial_speed, resistance_type, gravity=9.81):
    """
    Система уравнений движения камня
    state = [x, y, vx, vy] - положение и скорость
    """
    x, y, vx, vy = state

    speed = np.sqrt(vx ** 2 + vy ** 2)

    if resistance_type == 1:
        if resistance_coef == 0:
            ax = 0
            ay = -gravity
        else:
            ax = - (resistance_coef / mass) * vx
            ay = -gravity - (resistance_coef / mass) * vy
    else:
        if resistance_coef == 0:
            ax = 0
            ay = -gravity
        else:
            ax = - (resistance_coef / mass) * vx * speed
            ay = -gravity - (resistance_coef / mass) * vy * speed

    return [vx, vy, ax, ay]


def calculate_trajectory(mass, angle_deg, initial_speed, resistance_coef, resistance_type, max_time=20, time_step=0.01):
    """
    Расчет траектории камня
    """

    angle_rad = np.radians(angle_deg)

    vx0 = initial_speed * np.cos(angle_rad)
    vy0 = initial_speed * np.sin(angle_rad)
    initial_state = [0, 0, vx0, vy0]

    time_points = np.arange(0, max_time, time_step)

    solution = odeint(stone_motion, initial_state, time_points,
                      args=(mass, resistance_coef, angle_deg, initial_speed, resistance_type))

    x = solution[:, 0]
    y = solution[:, 1]

    ground_indices = np.where(y < 0)[0]
    if len(ground_indices) > 0:
        first_negative_index = ground_indices[0]
        x = x[:first_negative_index]
        y = y[:first_negative_index]

    y = np.maximum(y, 0)

    if len(x) > 0:
        flight_distance = x[-1]
        max_height = np.max(y)
    else:
        flight_distance, max_height = 0, 0

    return x, y, flight_distance, max_height


def run_tests():
    """
        Автотесты по вашей просьбе, Даниил Петрович
    """
    print("Запуск автотестов")

    test_cases = [
        (1, 45, 10, 0, 1, "Нет сопротивления (вязкое трение = 0)"),
        (1, 45, 10, 0.1, 1, "Вязкое трение"),
        (1, 45, 10, 0, 2, "Нет сопротивления (лобовое = 0)"),
        (1, 45, 10, 0.01, 2, "Лобовое сопротивление"),
        (0.5, 30, 15, 0.05, 1, "Меньшая масса"),
    ]

    for i, (mass, angle, speed, res_coef, res_type, desc) in enumerate(test_cases, 1):
        try:
            x, y, distance, max_height = calculate_trajectory(mass, angle, speed, res_coef, res_type)
            print(f"Тест {i}: {desc}")
            print(f"  Дальность: {distance:.2f} м, Высота: {max_height:.2f} м")

            if distance < 0 or max_height < 0:
                print("Ошибка: отрицательные значения!")
            elif distance > 1000 or max_height > 1000:
                print("Предупреждение: очень большие значения")
            else:
                print("Тест пройден")

        except Exception as e:
            print(f"Ошибка в тесте {i}: {e}")

        print()


def main():
    """
    Моделирование полета камня
    """

    try:
        mass = float(input("Масса камня (кг): "))
        angle = float(input("Угол броска (градусы): "))
        initial_speed = float(input("Начальная скорость (м/с): "))

        if angle < -360 or angle > 360:
            raise ValueError("Угол должен быть в диапазоне от -360 до 360 градусов")

        print("\nТип сопротивления:")
        print("1 - Вязкое трение")
        print("2 - Лобовое сопротивление")
        resistance_type = int(input("Выберите (1 или 2): "))

        if resistance_type not in [1, 2]:
            raise ValueError("Тип сопротивления должен быть 1 или 2")

        resistance_coef = float(input("Коэффициент сопротивления: "))


        if mass <= 0:
            raise ValueError("Масса должна быть положительной")
        if initial_speed <= 0:
            raise ValueError("Начальная скорость должна быть положительной")
        if resistance_coef < 0:
            raise ValueError("Коэффициент сопротивления не может быть отрицательным")

    except ValueError as e:
        print(f"Ошибка: {e}")
        return

    max_time = max(20, initial_speed / 10)

    x, y, distance, max_height = calculate_trajectory(
        mass, angle, initial_speed, resistance_coef, resistance_type, max_time)

    print(f"\n=== Результаты ===")
    print(f"Дальность полета: {distance:.2f} м")
    print(f"Максимальная высота: {max_height:.2f} м")

    if len(x) > 0 and distance > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'b-', linewidth=2, label='Траектория камня')
        plt.plot([0, distance], [0, 0], 'k-', alpha=0.3, linewidth=1)
        plt.plot(distance, 0, 'ro', markersize=8, label='Точка падения')
        plt.plot(x[np.argmax(y)], max_height, 'go', markersize=6, label='Максимальная высота')

        plt.xlabel('Расстояние, м')
        plt.ylabel('Высота, м')
        plt.title('Траектория полета камня')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis('equal')
        plt.show()
    else:
        print("Невозможно построить график: траектория отсутствует")


if __name__ == '__main__':
    run_tests()

    print("Ввод данных")
    main()
