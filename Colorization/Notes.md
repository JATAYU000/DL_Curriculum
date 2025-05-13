
[Colorful Image Colorization](https://richzhang.github.io/colorization/resources/colorful_eccv2016.pdf)

## **Abstract**
Given the lightness channel L, our system predicts the corresponding a and
b color channels of the image in the CIE Lab colorspace
![[Pasted image 20250508100714.png]]

## **Approach**
![[Pasted image 20250508101616.png]]

### **Objective Function**
With L your are predicting Y which is (a,b) so naturally L2 euclidean loss between predicted and ground truth colors
![[Pasted image 20250508102200.png]]
**Problem** with this is it not robust with multimodal nature of colorization problem say apple is green and red etc,so Euclidean loss will be the mean of the set. In color prediction, this averaging effect favors grayish, desaturated results
**Solution** is to make it multinomial classification change continous (a,b) to discrete bins of 10 grid size , and we use multinomial cross enthropy loss compares predicted distribution to the soft ground truth distribution, encourages high prob mass near the ground truth labels/bins/colors.
ie 313 class probabilty distribution.
![[Pasted image 20250508103734.png]]
Imagine dividing the fig b in grid size of 10 x 10 and there would be 313 valid ones. that is essentially what we are doing.

### **Class Re-balancing** and **Point Estimates**
In Natural Images there is more low ab values like sky, pavement,dirt etc. So training will be biased towards the desaturated colours.

Instead of One hot encoding the bin labels they find 5 nearest bins they assign weights to these bins using Gaussian function based on distance. hence making it forgiving.
![[Pasted image 20250508152041.png]]
Now from Z (prob distribution for 313 labels) find Y which is single (a,b) 
**Annealing** -> take log of each prob Zi -> divide by T (temperature eg = 0.38) and exp back -> Normalise again. mean of this **annealing** result will be Y = (a,b) value. only PROBLEM is that it is **NOT differentiable**
![[Pasted image 20250508181907.png]]
Note:
- Anealing mean is not differentiable
- H is just a simple computation with const T, so doesn;t need backward pass, just a deterministic math operation
- T = 1 (mean), T = .38 (Annealed mean), T -> 0 (mode)

### **Evaluation**
Our full method, with classification loss, defined in Equation 2,
and class rebalancing, as described in Section 2.2. The network was trained (1.3 M images from ImageNet) from scratch with k-means initialization , using the ADAM solver for approximately 450k iterations.


## **Implementation**

[GitHub Implementation](https://github.com/richzhang/colorization)