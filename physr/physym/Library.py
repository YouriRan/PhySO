import warnings as warnings
import numpy as np

# Internal imports
from physr.physym import Functions as Func
from physr.physym import Token as Tok

class Library:
    """
        Object containing choosable tokens and their properties for a task of symbolic computation
        (non_positional and semi_positional).
        See Token.__init__ for full description of token properties.

        Attributes
        ----------
        lib_tokens : numpy.array of Token.Token
            List of tokens in the library.
        terminal_units_provided : bool
            Were all terminal tokens (arity = 0) units constraints (choosable and parents only) provided ?, ie is it
            possible to compute units constraints.
        n_library : int
            Number of tokens in the library (including superparent).
        n_choices  : int
            Number of choosable tokens (does not include superparent placeholder).
        superparent : Token.Token
            Placeholder token representing the output symbolic function (eg. y in y = 2*x + 1).
        dummy : Token.Token
            Placeholder token to complete program trees during program generation.
        invalid : Token.Token
            Placeholder for tokens that are not yet generated.
        lib_name_to_idx : dict of {str : int}
            Dictionary containing names and corresponding token index in the library.

        lib_name : numpy.array of str
        lib_choosable_name : numpy.array of str
        lib_sympy_repr : numpy.array of str
        lib_function : numpy.array of objects (callable or None)
        properties : Token.VectTokens

        arity                     :  int
        complexity                :  float
        is_input_var              :  bool
        input_var_id              :  int
        behavior_id               :  int
        is_power                  :  bool
        power                     :  float

        is_constraining_phy_units :  bool
        phy_units                 :  numpy.array of shape (UNITS_VECTOR_SIZE,) of float

        Methods
        -------

        reset_library
            Resets token properties vectors based on current library of tokens list (lib_tokens).
        append_custom_tokens
            Adding list of custom Tokens to library.
        append_tokens_from_names
            Creates tokens by passing arguments to Functions.make_tokens and adding them to library.

        Examples
        --------
        add = Tok.Token(name='add', sympy_repr='add', arity=2, complexity=0, is_input_var=False,
                        function=np.add,
                        input_var_id=None)
        args_make_tokens = {
            # operations
            "op_names"             : ["mul", "neg", "inv", "sin"],
            "use_protected_ops"    : False,
            # input variables
            "input_var_ids"        : {"x" : 0         , "v" : 1          , "t" : 2,        },
            "input_var_units"      : {"x" : [1, 0, 0] , "v" : [1, -1, 0] , "t" : [0, 1, 0] },
            "input_var_complexity" : {"x" : 0.        , "v" : 1.         , "t" : 0.,       },
            # constants
            "constants"            : {"pi" : np.pi     , "c" : 3e8       , "M" : 1e6       },
            "constants_complexity" : {"pi" : 0.        , "c" : 0.        , "M" : 1.        },
                           }
        my_lib = Lib.Library(custom_tokens = [add,], args_make_tokens = args_make_tokens,
                     superparent_units = [1,-1,0], superparent_name = "v")

    """

    def __init__(self, custom_tokens = None, args_make_tokens = None, superparent_units = None, superparent_name = "y"):
        """
        Defines choosable tokens in the library.
        Parameters
        ----------
        args_make_tokens : dict or None
            If not None, arguments are passed to Functions.make_tokens and tokens are added to the library.
        custom_tokens : list of Token.Token or None
            If not None, the tokens are added to the library.
        superparent_units : array_like of float
            Physical units vector of output symbolic function (eg. [1, 0, 0] for a distance assuming a convention such
            as [m, s, kg,...]). None if not using physical units. None by default.
        superparent_name : str
            Customizable name of output symbolic function for display purposes. "y" by default.
        """
        # ------------------------------ SUPERPARENT ------------------------------
        # Should happen 1st so that superparent.name is defined when calling reset_library
        # Units # using retrieve_units for its error raising + padding features
        if superparent_units is None:
            superparent_units_dict = None  # will result in y_units = None => Token.units = vector of np.NAN
        else:
            superparent_units_dict = {superparent_name: superparent_units}
        y_is_constraining_phy_units, y_units = Func.retrieve_units(superparent_units_dict, superparent_name)
        # Function
        def superparent_func():
            raise ValueError("Superparent is a placeholder, it should never be called")
        # Token
        self.superparent = Tok.Token(
            name                      = superparent_name,
            sympy_repr                = superparent_name,
            arity                     = 0,
            complexity                = 0.,
            is_input_var              = False,
            function                  = superparent_func,
            is_constraining_phy_units = y_is_constraining_phy_units,
            phy_units                 = y_units,
        )

        # ------------------------------ DUMMY ------------------------------
        # Function
        def dummy_func():
            raise ValueError("Dummy is a placeholder, it should never be called")
        # Token
        self.dummy = Tok.Token(
            name                      = Tok.DUMMY_TOKEN_NAME,
            sympy_repr                = Tok.DUMMY_TOKEN_NAME,
            arity                     = 0,
            complexity                = 0.,
            is_input_var              = False,
            function                  = dummy_func,
        )

        # ------------------------------ INVALID ------------------------------
        # Function
        def invalid_func():
            raise ValueError("Invalid is a placeholder, it should never be called")
        # Token
        self.invalid = Tok.Token(
            name                      = Tok.INVALID_TOKEN_NAME,
            sympy_repr                = Tok.INVALID_TOKEN_NAME,
            arity                     = 0,
            complexity                = 0.,
            is_input_var              = False,
            function                  = invalid_func,
        )

        # ------------------------------ PLACEHOLDERS ------------------------------
        self.placeholders = [self.superparent, self.dummy, self.invalid]

        # ------------------------------ LIB OF TOKENS ------------------------------
        self.choosable_tokens = []
        self.append_tokens_from_names(args_make_tokens = args_make_tokens)
        self.append_custom_tokens(custom_tokens = custom_tokens)

        # ------------------------------ INPUT VAR IDS ------------------------------
        # Ids of input variables available
        self.input_var_ids = self.input_var_id[self.is_input_var]

    def reset_library(self):
        self.lib_tokens = np.array(self.choosable_tokens + self.placeholders)
        self.assert_units()
        # Number of tokens in the library
        self.n_library = len(self.lib_tokens)
        self.n_choices = len(self.choosable_tokens)
        # Idx of placeholders
        self.superparent_idx = self.n_choices + 0
        self.dummy_idx       = self.n_choices + 1
        self.invalid_idx     = self.n_choices + 2
        # Token representation
        self.lib_name           = np.array([token.name for token in self.lib_tokens       ]).astype(str)  # str (<MAX_NAME_SIZE) )
        self.lib_choosable_name = np.array([token.name for token in self.choosable_tokens ]).astype(str)  # str (<MAX_NAME_SIZE) )
        self.lib_sympy_repr     = np.array([token.sympy_repr for token in self.lib_tokens ]).astype(str)  # str (<MAX_NAME_SIZE) )
        # Object properties
        self.lib_function   = np.array([token.function   for token in self.lib_tokens])  # object (callable or None)
        # Vectorized properties
        self.properties = Tok.VectTokens(shape = (1, self.n_library,), invalid_token_idx = self.invalid_idx) # not using positional properties
        self.properties.arity                     [0, :] = np.array([token.arity                     for token in self.lib_tokens]).astype(int  )  # int
        self.properties.complexity                [0, :] = np.array([token.complexity                for token in self.lib_tokens]).astype(float)  # float
        self.properties.is_input_var              [0, :] = np.array([token.is_input_var              for token in self.lib_tokens]).astype(bool )  # bool
        self.properties.input_var_id              [0, :] = np.array([token.input_var_id              for token in self.lib_tokens]).astype(int  )  # int
        self.properties.behavior_id               [0, :] = np.array([token.behavior_id               for token in self.lib_tokens]).astype(int  )  # int
        self.properties.is_power                  [0, :] = np.array([token.is_power                  for token in self.lib_tokens]).astype(bool )  # bool
        self.properties.power                     [0, :] = np.array([token.power                     for token in self.lib_tokens]).astype(float)  # float
        self.properties.is_constraining_phy_units [0, :] = np.array([token.is_constraining_phy_units for token in self.lib_tokens]).astype(bool )  # bool
        self.properties.phy_units                 [0, :] = np.array([token.phy_units                 for token in self.lib_tokens]).astype(float)  # float
        # Giving access to vectorized properties to user without having to use [0, :] at each property access
        self.arity                     = self.properties.arity                     [0, :]
        self.complexity                = self.properties.complexity                [0, :]
        self.is_input_var              = self.properties.is_input_var              [0, :]
        self.input_var_id              = self.properties.input_var_id              [0, :]
        self.behavior_id               = self.properties.behavior_id               [0, :]
        self.is_power                  = self.properties.is_power                  [0, :]
        self.power                     = self.properties.power                     [0, :]
        self.is_constraining_phy_units = self.properties.is_constraining_phy_units [0, :]
        self.phy_units                 = self.properties.phy_units                 [0, :]
        # Helper dict
        self.lib_name_to_idx             = {self.lib_name[i] : i                  for i in range (self.n_library)}
        self.lib_choosable_name_to_idx   = {self.lib_name[i] : i                  for i in range (self.n_choices)}
        self.lib_name_to_token           = {self.lib_name[i] : self.lib_tokens[i] for i in range (self.n_library)}

    def append_custom_tokens(self, custom_tokens = None):
        # ----- Handling custom tokens -----
        if custom_tokens is None:
            custom_tokens = []
        self.choosable_tokens += custom_tokens
        self.reset_library()

    def append_tokens_from_names(self, args_make_tokens = None):
        # ----- Handling created tokens -----
        if args_make_tokens is None:
            created_tokens = []
        else:
            created_tokens = Func.make_tokens(**args_make_tokens).tolist()
        self.choosable_tokens += created_tokens
        self.reset_library()

    def assert_units(self):
        """
        Checks if all terminal tokens (arity = 0) have units constraints ie if units constraints can be computed.
        Tokens in library come from various units assignments processes (from make_tokens : operation definition in
        Functions, input_var_units dict, constants_units dict ; from custom tokens ; superparent units in
        Library.__init__)
        """
        self.terminal_units_provided = True
        # Checking all tokens (except dummy and valid which have arity = 0 and free dim)
        for token in self.choosable_tokens + [self.superparent]:
            if token.arity == 0 and token.is_constraining_phy_units == False:
                # is_constraining_phy_units = False <=> phy_units is Free ie phy_units = vector of NAN
                # (this is ensured via exceptions in Token.Token.__init__)
                warnings.warn("The units of token %s were not provided (is_constraining_phy_units=%s ; phy_units=%s), "
                              "unable to compute units constraints."
                              %(token.name, token.is_constraining_phy_units, token.phy_units))
                self.terminal_units_provided = False
        return None

    def get_choosable_prop(self, attr):
        """
        Returns vectorized property of choosable tokens of the library.
        Parameters
        ----------
        attr : str
            Vectorized token property.
        Returns
        -------
        property : numpy.array of shape (self.n_choices,) of type ?
            ? depends on the property
        """
        return getattr(self.properties, attr)[0][:self.n_choices]

    def __repr__(self):
        return str(self.lib_tokens)

    def __getitem__(self, item):
        return self.lib_tokens[item]