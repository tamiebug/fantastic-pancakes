import unittest
import fnmatch
import importlib
import os


def caffeIsPresent():
    """ Attempts to import Caffe.  Returns true if it is possible """
    try:
        import caffe
    except ImportError:
        return False

    return True


def directoriesToModuleNames(dirNames):
    """ Converts a directory name into a module name """
    moduleNames = ["test." + name for name in dirNames]
    moduleNames = [name.replace(".py", "") for name in moduleNames]
    return moduleNames


def main():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    test_files = []
    for root, dirnames, filenames in os.walk(this_dir):
        for filename in fnmatch.filter(filenames, '*_test.py'):
            test_files.append(os.path.join(root, filename))
    if not caffeIsPresent():
        test_files = [test for test in test_files if 'caffe' not in test]

    test_files = [test for test in test_files if 'smoke' not in test]

    test_files = [os.path.relpath(name, this_dir) for name in test_files]
    test_module_names = directoriesToModuleNames(test_files)
    test_modules = [importlib.import_module(name) for name in test_module_names]

    testSuite = unittest.TestSuite()

    for module in test_modules:
        testSuite.addTest(unittest.defaultTestLoader.loadTestsFromModule(module))

    print("About to run {} tests".format(testSuite.countTestCases()))

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    testRunner = unittest.TextTestRunner(verbosity=2, buffer=True)
    testRunner.run(testSuite)
    return


if __name__ == "__main__":
    main()
