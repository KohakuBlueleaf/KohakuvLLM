//! JSON-schema token masks for vLLM (`kohakuvllm.grammar`): `schema` compiles a schema, `matcher`
//! walks reply bytes, `store` memoises one mask per distinct clamped state.

pub mod matcher;
pub mod schema;
pub mod store;

#[cfg(feature = "python")]
mod py {
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::PyBytes;

    use crate::store;

    #[pyclass(module = "kohakuvllm_grammar")]
    struct Store(store::Store);

    #[pymethods]
    impl Store {
        /// tokens: each token's bytes (None: never allowed), eos: end-of-sequence ids,
        /// vocab: the logits' width, max_ws: whitespace bytes allowed per gap
        #[new]
        fn new(tokens: Vec<Option<Vec<u8>>>, eos: Vec<u32>, vocab: usize, max_ws: u8) -> Self {
            Store(store::Store::new(tokens, eos, vocab, max_ws))
        }

        fn compile(&mut self, schema: &str) -> PyResult<u32> {
            self.0.compile(schema).map_err(PyValueError::new_err)
        }

        fn matcher(&mut self, grammar: u32) -> u32 {
            self.0.matcher(grammar)
        }

        fn release(&mut self, handle: u32) {
            self.0.release(handle)
        }

        fn step(&mut self, py: Python<'_>, handles: Vec<u32>, tokens: Vec<Vec<u32>>) -> Vec<u32> {
            py.allow_threads(|| self.0.step(&handles, &tokens))
        }

        /// The words of every mask added since the last call (native-endian uint32)
        fn take_new<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyBytes> {
            let words = self.0.take_new();
            let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_ne_bytes()).collect();
            PyBytes::new(py, &bytes)
        }

        fn mask<'py>(&self, py: Python<'py>, id: u32) -> Bound<'py, PyBytes> {
            let bytes: Vec<u8> = self.0.mask(id).iter().flat_map(|w| w.to_ne_bytes()).collect();
            PyBytes::new(py, &bytes)
        }

        #[getter]
        fn words(&self) -> usize {
            self.0.words
        }

        #[getter]
        fn count(&self) -> usize {
            self.0.count()
        }

        /// masks built, seconds building them, step calls, seconds in steps, tokens, rejected
        fn stats(&self) -> (u64, f64, u64, f64, u64, u64) {
            let s = self.0.stats;
            (s.masks, s.mask_s, s.steps, s.step_s, s.tokens, s.rejected)
        }
    }

    #[pymodule]
    fn kohakuvllm_grammar(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<Store>()?;
        m.add("EOS_ONLY", store::EOS_ONLY)?;
        Ok(())
    }
}
