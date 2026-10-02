# Third-party components

FARO calls the upstream AutoPGD implementation from the **AutoAttack** package by
Francesco Croce and Matthias Hein. Its source is not vendored or renamed here.
Installation obtains it from:

https://github.com/fra31/auto-attack

Pinned revision: `a39220048b3c9f2cca9a4d3a54604793c68eca7e`.
The upstream repository's LICENSE at this revision contains the following MIT
notice (the repository's license text is used rather than its setup classifier):

> MIT License
>
> Copyright (c) 2020 Francesco Croce
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

TensorFlow, NumPy and PyTorch are external dependencies with their own license
terms. No third-party classifier weights or dataset files are redistributed in
this repository. Historical benchmark names identify the evaluated checkpoints;
they do not imply endorsement by their authors.
