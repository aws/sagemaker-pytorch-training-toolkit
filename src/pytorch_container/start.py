from container_support import ContainerSupport
import training

cs = ContainerSupport()
cs.register_engine(training.engine)

if __name__ == '__main__':
    cs.run()
