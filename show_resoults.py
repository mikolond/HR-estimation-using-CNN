from matplotlib import pyplot as plt
import numpy


def draw_gauss(ax, mu, theta):
    '''Draws gauss function into the existing figure plot'''
    x = numpy.linspace(mu - 3*theta, mu + 3*theta, 100)
    y = (1/(theta * numpy.sqrt(2 * numpy.pi))) * numpy.exp(-0.5 * ((x - mu) / theta)**2)
    ax.plot(x, y)


def main():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    mu = 35.05
    theta = 20.8
    draw_gauss(ax, mu, theta)
    mu = 33.06
    theta = 20.47
    draw_gauss(ax, mu, theta)
    mu = 29.6
    theta = 27.5
    draw_gauss(ax, mu, theta)

    # mu = 14.52
    # theta = 22.24
    # draw_gauss(ax, mu, theta)


    plt.show()

if __name__ == "__main__":
    main()

