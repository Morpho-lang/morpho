# Contributing to Morpho
Hello there! It's great to see your interest in improving Morpho!

We welcome contributions from everyone. If you are unsure of anything, feel free to reach out via the Github, submit an issue or make a pull request.

There are many ways you can contribute to Morpho:

* If you have identified a bug, you can report it by [submitting an issue with the `Bug` label](https://github.com/Morpho-lang/morpho/issues/new?assignees=&labels=bug%2C+Needs+Priority&template=bug_report.md&title=%5BBug%5D). If you have solved a bug, first off, that's excellent, and thank you! You can submit your fix as a pull request to the [dev branch](https://github.com/Morpho-lang/morpho/tree/dev) of Morpho. This branch is where the updates are collected before eventually releasing them into a new version in the main branch. All pull requests to `dev` are passed through the automated testing suite. Check [below](#unit-tests) for more on that and about adding your own tests!

* If you use Morpho but are new to GitHub, or to contributing to Morpho, the issues labeled [`good first issue`](https://github.com/Morpho-lang/morpho/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) highlight easy-to-fix bugs that will get you started. [Here](https://gist.github.com/Chaser324/ce0505fbed06b947d962) is a guide that explains the best practices for making a pull request.

* If you want to propose a new feature in Morpho, you can submit an issue with the [`enhancement`](https://github.com/Morpho-lang/morpho/issues?q=is%3Aissue+is%3Aopen+label%3Aenhancement) label.

* Morpho is highly modular and modules providing new features are especially welcome.

* Help with unit tests, additional documentation etc. are also great ways to contribute to the project.

All contributors are expected to follow the [Morpho Code of Conduct](https://github.com/Morpho-lang/morpho/blob/main/CODE_OF_CONDUCT.md).

For further guidance and pointers, a developer's guide gradually being assembled [devguide](https://github.com/Morpho-lang/morpho/blob/main/devguide/devguide.pdf). In the meantime, we encourage you to [join our Slack community](https://join.slack.com/t/morphoco/shared_invite/zt-1hiby4iqv-UhqKEeqZih0vSG3k4gEfXQ) or get in touch via email.

## Unit-tests

Morpho has an extensive set of unit-tests to make sure any new piece of code doesn't break essential functionality. Moreover, code within any pull requests to `dev` or `main` is automatically put through the test suite. While this will catch failing tests, if any, you can make sure all the tests are passing on your branch beforehand by running the test suite locally:

    cd test
    python3 test.py

If you have fixed a new bug, chances are the existing unit-tests didn't capture that buggy behavior. In that case, it's a good idea to _add_ new tests to lock down the behavior. You can add tests simply by adding the test file anywhere under the `test/` directory. The files are organized around topic for convenience, but any file therein will get tested.

We highly welcome contributions to the testing suite. Try writing tests that don't overlap with the existing tests, and help us lock down any remaining bugs in Morpho's functionality.

### Formatting a unit-test

While a new unit-testing module is [in the works](https://github.com/Morpho-lang/morpho/pull/147), the current unit-test are executed in `python` by looking for the keyword `expect`. For instance, here is an example from the test `power.morpho` that tests the arithmetic power operator:

    // Test power operator
    print 2^2
    // expect: 4

Here, the operation is performed and the output is printed. The special comment that starts with `// expect: ` followed by the expected output is picked up by the python testing file and compared with the output. For multiple `print` statements, the `// expect: ` comments can appear anywhere in the code, as long as they are in the right order. Other comments work as regular comments and can be used to annotate the test.
