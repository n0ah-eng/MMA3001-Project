"""sensorgap - reconstruction of missing indoor environmental sensor data.

MMA3001 Numerical Methods and Machine Learning, individual project.

Modules:
    io: load raw Monash Smart Infrastructure exports into tidy tables.
    cleaning: de-duplicate, bound-check, audit and regularise sensor data.
    interpolation: univariate gap-filling methods (linear, cubic, PCHIP, Akima).
    validation: mask-and-reconstruct experiments and error metrics.
"""

__version__ = "0.1.0"
