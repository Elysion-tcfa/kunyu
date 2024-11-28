#include <torch/extension.h>
#include <shtns.h>
#ifdef USE_GPU
#include <shtns_cuda.h>
#define DEFAULT_FLAGS (SHT_THETA_CONTIGUOUS | SHT_ALLOW_GPU)
#else
#define DEFAULT_FLAGS (SHT_THETA_CONTIGUOUS)
#endif

struct SHT {
	SHT(int lmax, int mmax, int mres, int norm) {
		cfg = shtns_create(lmax, mmax < 0 ? lmax / mres : mmax, mres, (shtns_norm)norm);
		batch_size = 1;
	}

	~SHT() {
		shtns_destroy(cfg);
	}

	void set_batch(int new_batch_size) {
		batch_size = (size_t)new_batch_size;
		shtns_set_many(cfg, new_batch_size, 0);
	}

	void set_grid(int nlat, int nphi, int flags, double eps) {
		shtns_set_grid(cfg, (shtns_type)(flags | DEFAULT_FLAGS), eps, nlat, nphi);
		gpu_use_float = flags & SHT_FP32;
	}

	void print() {
		shtns_print_cfg(cfg);
	}

	int idx(int l, int m) {
		if (l > cfg->lmax || m > l || m < 0 || m > cfg->mmax * cfg->mres || m % cfg->mres != 0)
			throw std::runtime_error("invalid l or m");
		return l + (((m / cfg->mres) * (2 * cfg->lmax + 2 - (m + cfg->mres))) >> 1);
	}

	torch::Tensor cos_theta() {
		torch::Tensor ret = torch::empty({cfg->nlat}, torch::TensorOptions().dtype(torch::kFloat64));
		double *ptr = ret.data_ptr<double>();
		memcpy(ptr, cfg->ct, sizeof(double) * cfg->nlat);
		return ret;
	}

	torch::Tensor gauss_weights() {
		torch::Tensor ret = torch::empty({cfg->nlat_2}, torch::TensorOptions().dtype(torch::kFloat64));
		double *ptr = ret.data_ptr<double>();
		if (shtns_gauss_wts(cfg, ptr) == 0)
			throw std::runtime_error("not a gauss grid");
		return ret;
	}

	torch::Tensor grid_to_sh(torch::Tensor grid) {
		std::vector<int64_t> in_shape = grid.sizes().vec();
		if (in_shape.size() < 2 + (batch_size > 1))
			throw std::runtime_error("input should have shape (..., batch, nphi, nlat) if batch size is greater than 1 or (..., nphi, nlat) if batch size is 1");
		bool has_batch_dim = batch_size > 1;
		if (in_shape[in_shape.size() - 2] != cfg->nphi || in_shape[in_shape.size() - 1] != cfg->nlat ||
				(has_batch_dim && in_shape[in_shape.size() - 3] != batch_size))
			throw std::runtime_error("input should have shape (..., batch, nphi, nlat) if batch size is greater than 1 or (..., nphi, nlat) if batch size is 1");
		size_t batch = 1;
		for (size_t i = 0; i < in_shape.size() - 2 - (int)has_batch_dim; i++)
			batch *= in_shape[i];
		auto orig_type = grid.scalar_type();
		bool on_gpu = false;
		bool use_float = false;
#ifdef USE_GPU
		if (grid.device().type() == torch::DeviceType::CUDA)
			on_gpu = true;
		else
#endif
			if (grid.device().type() != torch::DeviceType::CPU)
				throw std::runtime_error("unsupported device");
		if (on_gpu && gpu_use_float) {
			grid = grid.contiguous().to(torch::kFloat32, false, on_gpu);
			use_float = true;
		} else
			grid = grid.contiguous().to(torch::kFloat64, false, on_gpu);
		void *in = use_float ? (void*)grid.data_ptr<float>() : (void*)grid.data_ptr<double>();
		std::vector<int64_t> out_shape(in_shape.begin(), in_shape.end() - 2 - (int)has_batch_dim);
		if (has_batch_dim) out_shape.emplace_back(batch_size);
		out_shape.emplace_back(cfg->nlm);
		torch::Tensor sh = torch::empty(out_shape, torch::TensorOptions().dtype(use_float ? torch::ScalarType::ComplexFloat : torch::ScalarType::ComplexDouble).device(grid.device()));
		void *out = (void*)sh.data_ptr();
		size_t in_inc = (size_t)(cfg->nlat * cfg->nphi) * batch_size, out_inc = (size_t)cfg->nlm * batch_size;
		for (size_t i = 0; i < batch; i++) {
#ifdef USE_GPU
			if (on_gpu)
				if (use_float)
					cu_spat_to_SH_float(cfg, (float*)in, (cplx_f*)out, cfg->lmax);
				else
					cu_spat_to_SH(cfg, (double*)in, (cplx*)out, cfg->lmax);
			else
#endif
				spat_to_SH(cfg, (double*)in, (cplx*)out);
			if (use_float) {
				in = (void*)((float*)in + in_inc);
				out = (void*)((cplx_f*)out + out_inc);
			} else {
				in = (void*)((double*)in + in_inc);
				out = (void*)((cplx*)out + out_inc);
			}
		}
		return orig_type == torch::ScalarType::Float ? sh.to(torch::ScalarType::ComplexFloat) : sh.to(torch::ScalarType::ComplexDouble);
	}

	torch::Tensor sh_to_grid(torch::Tensor sh) {
		std::vector<int64_t> in_shape = sh.sizes().vec();
		if (in_shape.size() < (1 + batch_size > 1))
			throw std::runtime_error("input should have shape (..., batch, nlm) or (..., nlm) if batch size is 1");
		bool has_batch_dim = batch_size > 1;
		if (in_shape[in_shape.size() - 1] != cfg->nlm || (has_batch_dim && in_shape[in_shape.size() - 2] != batch_size))
			throw std::runtime_error("incorrect size of last dimension");
		size_t batch = 1;
		for (size_t i = 0; i < in_shape.size() - 1 - (int)has_batch_dim; i++)
			batch *= in_shape[i];
		auto orig_type = sh.scalar_type();
		bool on_gpu = false;
		bool use_float = false;
#ifdef USE_GPU
		if (sh.device().type() == torch::DeviceType::CUDA)
			on_gpu = true;
		else
#endif
			if (sh.device().type() != torch::DeviceType::CPU)
				throw std::runtime_error("unsupported device");
		if (on_gpu && gpu_use_float) {
			sh = sh.contiguous().to(torch::ScalarType::ComplexFloat, false, on_gpu);
			use_float = true;
		} else
			sh = sh.contiguous().to(torch::ScalarType::ComplexDouble, false, on_gpu);
		void *in = (void*)sh.data_ptr();
		std::vector<int64_t> out_shape(in_shape.begin(), in_shape.end() - 1 - (int)has_batch_dim);
		if (has_batch_dim) out_shape.emplace_back(batch_size);
		out_shape.emplace_back(cfg->nphi);
		out_shape.emplace_back(cfg->nlat);
		torch::Tensor grid = torch::empty(out_shape, torch::TensorOptions().dtype(use_float ? torch::kFloat32 : torch::kFloat64).device(sh.device()));
		void *out = use_float ? (void*)grid.data_ptr<float>() : (void*)grid.data_ptr<double>();
		size_t in_inc = (size_t)cfg->nlm * batch_size, out_inc = (size_t)(cfg->nlat * cfg->nphi) * batch_size;
		for (size_t i = 0; i < batch; i++) {
#ifdef USE_GPU
			if (on_gpu)
				if (use_float)
					cu_SH_to_spat_float(cfg, (cplx_f*)in, (float*)out, cfg->lmax);
				else
					cu_SH_to_spat(cfg, (cplx*)in, (double*)out, cfg->lmax);
			else
#endif
				SH_to_spat(cfg, (cplx*)in, (double*)out);
			if (use_float) {
				in = (void*)((cplx_f*)in + in_inc);
				out = (void*)((float*)out + out_inc);
			} else {
				in = (void*)((cplx*)in + in_inc);
				out = (void*)((double*)out + out_inc);
			}
		}
		return orig_type == torch::ScalarType::ComplexFloat ? grid.to(torch::kFloat32) : grid.to(torch::kFloat64);
	}

	shtns_cfg cfg;
	size_t batch_size;
	bool gpu_use_float;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	pybind11::class_<SHT>(m, "SHT")
		.def(pybind11::init<int, int, int, int>(), pybind11::arg("lmax"), pybind11::arg("mmax") = -1, pybind11::arg("mres") = 1, pybind11::arg("norm") = (int)sht_orthonormal)
		.def("set_grid", &SHT::set_grid, pybind11::arg("nlat"), pybind11::arg("nphi"), pybind11::arg("flags") = (int)sht_quick_init, pybind11::arg("eps") = 1e-8)
		.def("set_batch", &SHT::set_batch, pybind11::arg("batch_size"))
		.def("print", &SHT::print)
		.def("idx", &SHT::idx)
		.def("cos_theta", &SHT::cos_theta)
		.def("gauss_weights", &SHT::gauss_weights)
		.def("grid_to_sh", &SHT::grid_to_sh)
		.def("sh_to_grid", &SHT::sh_to_grid)
		.def_property_readonly("lmax", [](const SHT &sh) { return sh.cfg->lmax; })
		.def_property_readonly("mmax", [](const SHT &sh) { return sh.cfg->mmax; })
		.def_property_readonly("mres", [](const SHT &sh) { return sh.cfg->mres; })
		.def_property_readonly("nlm", [](const SHT &sh) { return sh.cfg->nlm; })
		.def_property_readonly("nlat", [](const SHT &sh) { return sh.cfg->nlat; })
		.def_property_readonly("nphi", [](const SHT &sh) { return sh.cfg->nphi; })
		.def_property_readonly("batch_size", [](const SHT &sh) { return sh.batch_size; });
}
