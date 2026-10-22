//! A plugin named by an effect, a filter, or an adjustment.
//!
//! The project stores the id, the version, and the parameters. It does not
//! open the plugin and it does not store a shader.

use crate::Error;

/// A plugin id, a version, and the parameters the segment stores.
#[derive(Debug, Clone, PartialEq)]
pub struct Attachment {
    plugin: String,
    version: String,
    parameters: Vec<Parameter>,
}

/// One named value on an [`Attachment`].
#[derive(Debug, Clone, PartialEq)]
pub struct Parameter {
    name: String,
    value: ParameterValue,
}

/// A parameter value. A number is finite.
#[derive(Debug, Clone, PartialEq)]
pub enum ParameterValue {
    Number(f64),
    Bool(bool),
    Text(String),
}

impl Attachment {
    /// An attachment. The plugin id and the version must be non-empty.
    ///
    /// Parameter names must be non-empty and unique. A number must be finite.
    pub fn new(
        plugin: impl Into<String>,
        version: impl Into<String>,
        parameters: Vec<Parameter>,
    ) -> Result<Self, Error> {
        let plugin = plugin.into();
        let version = version.into();
        if plugin.is_empty() || version.is_empty() {
            return Err(Error::Plugin);
        }
        let mut names = Vec::with_capacity(parameters.len());
        for parameter in &parameters {
            if parameter.name.is_empty() || names.contains(&parameter.name.as_str()) {
                return Err(Error::Parameter);
            }
            if let ParameterValue::Number(number) = parameter.value {
                if !number.is_finite() {
                    return Err(Error::NonFinite);
                }
            }
            names.push(parameter.name.as_str());
        }
        Ok(Self {
            plugin,
            version,
            parameters,
        })
    }

    pub fn plugin(&self) -> &str {
        &self.plugin
    }

    pub fn version(&self) -> &str {
        &self.version
    }

    pub fn parameters(&self) -> &[Parameter] {
        &self.parameters
    }
}

impl Parameter {
    pub fn new(name: impl Into<String>, value: ParameterValue) -> Self {
        Self {
            name: name.into(),
            value,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn value(&self) -> &ParameterValue {
        &self.value
    }
}
