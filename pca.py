from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_and_center_dataset(filename):
    dataset = np.load(filename)
    ds_mean = np.mean(dataset, axis=0)
    ds_centered = dataset - ds_mean
    return ds_centered

def get_covariance(dataset):
    ds_transpose = np.transpose(dataset)
    ds_dot = np.dot(ds_transpose, dataset) # apply dot product
    ds_cov = ds_dot * (1 / (len(dataset)-1)) # divide by total number of images
    return ds_cov


def get_eig(S, m):
    size = len(S)
    vals, vecs = eigh(S, eigvals_only=False, subset_by_index=[size-m, size-1])
    ds_eigenvals = np.flip(np.diag(vals)) # make matrix diagonal and descending
    ds_eigenvecs = np.flip(vecs, axis=1)
    return ds_eigenvals, ds_eigenvecs


def get_eig_perc(S, perc):
    size = len(S)
    vals, vecs = get_eig(S, size) # get the eigenvalues and vectors
    sum = np.sum(vals) # get sum of eigenvalues
    thresh = perc * sum # eigenvalue threshold (find values greater than this)
    occurrences = vals > thresh # matrix of values over thresh
    num_vals = occurrences.sum() # count number over threshold
    ds_eigenvals, ds_eigenvecs = get_eig(S, num_vals)
    return ds_eigenvals, ds_eigenvecs 


def project_image(img, U):
    col_size = len(U[0]) # number of columns
    alpha_ij = np.zeros((col_size,)) # init zeros array of eigenvectorsx1 entries
    projected = np.zeros((len(U), col_size)) # init zeros array of dx1 entries
    for col in range(col_size): # dot product on each column
        alpha_ij[col] = np.dot(np.transpose(U[:, col]), img)
        # multiply alpha_ij with corresponding eigenvalue
        projected[:, col] = U[:, col] * alpha_ij[col]
    #print(alpha_ij)
    return np.sum(projected, axis=1)


def display_image(orig, proj):
    orig_reshape = np.reshape(orig, (32, 32)) # reshape image data
    proj_reshape = np.reshape(proj, (32, 32))
    figure, axes = plt.subplots(1, 2, figsize=(10, 3)) # setup plot
    axes[0].set_title('Original')
    axes[1].set_title('Projection')
    orig_img = axes[0].imshow(np.rot90(orig_reshape, 3), aspect='equal')
    proj_img = axes[1].imshow(np.rot90(proj_reshape, 3), aspect='equal')
    figure.colorbar(orig_img, ax=axes[0])
    figure.colorbar(proj_img, ax=axes[1])
    return plt.show()



