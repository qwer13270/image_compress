from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    x = np.load(filename)
    x = x - np.mean(x, axis=0)
    return x

def get_covariance(dataset):
    ans = np.cov(dataset.T)
    print(len(ans))
    print(len(ans[0]))
    return ans  

def get_eig(S, m):
    eigenvalues, eigenvectors = eigh(S)
    e_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors_sorted = eigenvectors[:,e_indices]
    eigenvalues_sorted = eigenvalues[e_indices]
    eigenvector_subset = eigenvectors_sorted[:,0:m]
    return eigenvalues_sorted[:m], eigenvector_subset

def get_eig_prop(S, prop):

    eigenvalues, eigenvectors = eigh(S)
    sum_eigenval = np.sum(eigenvalues)
    limit = sum_eigenval * prop
    new_eigenvalues, new_eigenvectors = eigh(S, subset_by_value = [limit, 10])

    pass

def project_image(image, U):
    num_columns = np.shape(U)[0]
    aij = np.dot(U.T, image)
    xpca = np.dot(U, aij)
    return xpca

def display_image(orig, proj):
    orig = np.reshape(orig, (32,32))
    proj = np.reshape(proj, (32,32))
    orig = np.rot90(orig, -1) # -1 means rotate 90 degrees counter-clockwise
    proj = np.rot90(proj, -1)
    fig, (Original, Projection) = plt.subplots(nrows = 1, ncols=2)
    org_show = Original.imshow(orig, aspect='equal')
    fig.colorbar(org_show, ax =Original)
    proj_show = Projection.imshow(proj, aspect='equal')
    fig.colorbar(proj_show, ax =Projection)
    plt.show()
    pass


x = load_and_center_dataset('YaleB_32x32.npy')
S = get_covariance(x)
Lambda, U = get_eig(S, 2)
test = np.array([1, 2, 3, 4])
projection = project_image(x[0], U)
display_image(x[0], projection)
print(type(x[0]))
print(projection)