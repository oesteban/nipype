# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""

Base I/O specifications for Nipype interfaces
.............................................

Define the API for the I/O of interfaces

"""
from __future__ import (print_function, division, unicode_literals,
                        absolute_import)

import os
from copy import deepcopy
from warnings import warn
from builtins import str, bytes
from packaging.version import Version

from ...external.due import due
from ...utils.filemanip import md5, hash_infile, hash_timestamp, to_str
from .traits_extension import (
    traits,
    Undefined,
    isdefined,
    TraitError,
    TraitDictObject,
    TraitListObject,
    has_metadata,
)

from ... import config, __version__

FLOAT_FORMAT = '{:.10f}'.format
nipype_version = Version(__version__)


class BaseTraitedSpec(traits.HasTraits):
    """
    Provide a few methods necessary to support nipype interface api

    The inputs attribute of interfaces call certain methods that are not
    available in traits.HasTraits. These are provided here.

    new metadata:

    * usedefault : set this to True if the default value of the trait should be
      used. Unless this is set, the attributes are set to traits.Undefined

    new attribute:

    * get_hashval : returns a tuple containing the state of the trait as a dict
      and hashvalue corresponding to dict.

    XXX Reconsider this in the long run, but it seems like the best
    solution to move forward on the refactoring.
    """
    package_version = nipype_version

    def __init__(self, **kwargs):
        """ Initialize handlers and inputs"""
        # NOTE: In python 2.6, object.__init__ no longer accepts input
        # arguments.  HasTraits does not define an __init__ and
        # therefore these args were being ignored.
        # super(TraitedSpec, self).__init__(*args, **kwargs)
        super(BaseTraitedSpec, self).__init__(**kwargs)
        traits.push_exception_handler(reraise_exceptions=True)
        undefined_traits = {}
        for trait in self.copyable_trait_names():
            if not self.traits()[trait].usedefault:
                undefined_traits[trait] = Undefined
        self.trait_set(trait_change_notify=False, **undefined_traits)
        self._generate_handlers()
        self.trait_set(**kwargs)

    def items(self):
        """ Name, trait generator for user modifiable traits
        """
        for name in sorted(self.copyable_trait_names()):
            yield name, self.traits()[name]

    def __repr__(self):
        """ Return a well-formatted representation of the traits """
        outstr = []
        for name, value in sorted(self.trait_get().items()):
            outstr.append('%s = %s' % (name, value))
        return '\n{}\n'.format('\n'.join(outstr))

    def _generate_handlers(self):
        """Find all traits with the 'xor' metadata and attach an event
        handler to them.
        """
        has_xor = dict(xor=lambda t: t is not None)
        xors = self.trait_names(**has_xor)
        for elem in xors:
            self.on_trait_change(self._xor_warn, elem)
        has_deprecation = dict(deprecated=lambda t: t is not None)
        deprecated = self.trait_names(**has_deprecation)
        for elem in deprecated:
            self.on_trait_change(self._deprecated_warn, elem)

    def _xor_warn(self, obj, name, old, new):
        """ Generates warnings for xor traits
        """
        if isdefined(new):
            trait_spec = self.traits()[name]
            # for each xor, set to default_value
            for trait_name in trait_spec.xor:
                if trait_name == name:
                    # skip ourself
                    continue
                if isdefined(getattr(self, trait_name)):
                    self.trait_set(
                        trait_change_notify=False, **{
                            '%s' % name: Undefined
                        })
                    msg = ('Input "%s" is mutually exclusive with input "%s", '
                           'which is already set') % (name, trait_name)
                    raise IOError(msg)

    def _deprecated_warn(self, obj, name, old, new):
        """Checks if a user assigns a value to a deprecated trait
        """
        if isdefined(new):
            trait_spec = self.traits()[name]
            msg1 = ('Input %s in interface %s is deprecated.' %
                    (name, self.__class__.__name__.split('InputSpec')[0]))
            msg2 = ('Will be removed or raise an error as of release %s' %
                    trait_spec.deprecated)
            if trait_spec.new_name:
                if trait_spec.new_name not in self.copyable_trait_names():
                    raise TraitError(msg1 + ' Replacement trait %s not found' %
                                     trait_spec.new_name)
                msg3 = 'It has been replaced by %s.' % trait_spec.new_name
            else:
                msg3 = ''
            msg = ' '.join((msg1, msg2, msg3))
            if Version(str(trait_spec.deprecated)) < self.package_version:
                raise TraitError(msg)
            else:
                if trait_spec.new_name:
                    msg += 'Unsetting old value %s; setting new value %s.' % (
                        name, trait_spec.new_name)
                warn(msg)
                if trait_spec.new_name:
                    self.trait_set(
                        trait_change_notify=False,
                        **{
                            '%s' % name: Undefined,
                            '%s' % trait_spec.new_name: new
                        })

    def trait_get(self, **kwargs):
        """ Returns traited class as a dict

        Augments the trait get function to return a dictionary without
        notification handles
        """
        out = super(BaseTraitedSpec, self).trait_get(**kwargs)
        out = self._clean_container(out, Undefined)
        return out

    get = trait_get

    def get_traitsfree(self, **kwargs):
        """ Returns traited class as a dict

        Augments the trait get function to return a dictionary without
        any traits. The dictionary does not contain any attributes that
        were Undefined
        """
        out = super(BaseTraitedSpec, self).trait_get(**kwargs)
        out = self._clean_container(out, skipundefined=True)
        return out

    def _clean_container(self, objekt, undefinedval=None, skipundefined=False):
        """Convert a traited obejct into a pure python representation.
        """
        if isinstance(objekt, TraitDictObject) or isinstance(objekt, dict):
            out = {}
            for key, val in list(objekt.items()):
                if isdefined(val):
                    out[key] = self._clean_container(val, undefinedval)
                else:
                    if not skipundefined:
                        out[key] = undefinedval
        elif (isinstance(objekt, TraitListObject) or isinstance(objekt, list) or
              isinstance(objekt, tuple)):
            out = []
            for val in objekt:
                if isdefined(val):
                    out.append(self._clean_container(val, undefinedval))
                else:
                    if not skipundefined:
                        out.append(undefinedval)
                    else:
                        out.append(None)
            if isinstance(objekt, tuple):
                out = tuple(out)
        else:
            out = None
            if isdefined(objekt):
                out = objekt
            else:
                if not skipundefined:
                    out = undefinedval
        return out

    def has_metadata(self, name, metadata, value=None, recursive=True):
        """
        Return has_metadata for the requested trait name in this
        interface
        """
        return has_metadata(
            self.trait(name).trait_type, metadata, value, recursive)

    def get_hashval(self, hash_method=None):
        """Return a dictionary of our items with hashes for each file.

        Searches through dictionary items and if an item is a file, it
        calculates the md5 hash of the file contents and stores the
        file name and hash value as the new key value.

        However, the overall bunch hash is calculated only on the hash
        value of a file. The path and name of the file are not used in
        the overall hash calculation.

        Returns
        -------
        list_withhash : dict
            Copy of our dictionary with the new file hashes included
            with each file.
        hashvalue : str
            The md5 hash value of the traited spec

        """
        list_withhash = []
        list_nofilename = []
        for name, val in sorted(self.trait_get().items()):
            if not isdefined(val) or self.has_metadata(name, "nohash", True):
                # skip undefined traits and traits with nohash=True
                continue

            hash_files = (not self.has_metadata(name, "hash_files", False) and
                          not self.has_metadata(name, "name_source"))
            list_nofilename.append((name,
                                    self._get_sorteddict(
                                        val,
                                        hash_method=hash_method,
                                        hash_files=hash_files)))
            list_withhash.append((name,
                                  self._get_sorteddict(
                                      val,
                                      True,
                                      hash_method=hash_method,
                                      hash_files=hash_files)))
        return list_withhash, md5(to_str(list_nofilename).encode()).hexdigest()

    def _get_sorteddict(self,
                        objekt,
                        dictwithhash=False,
                        hash_method=None,
                        hash_files=True):
        if isinstance(objekt, dict):
            out = []
            for key, val in sorted(objekt.items()):
                if isdefined(val):
                    out.append((key,
                                self._get_sorteddict(
                                    val,
                                    dictwithhash,
                                    hash_method=hash_method,
                                    hash_files=hash_files)))
        elif isinstance(objekt, (list, tuple)):
            out = []
            for val in objekt:
                if isdefined(val):
                    out.append(
                        self._get_sorteddict(
                            val,
                            dictwithhash,
                            hash_method=hash_method,
                            hash_files=hash_files))
            if isinstance(objekt, tuple):
                out = tuple(out)
        else:
            out = None
            if isdefined(objekt):
                if (hash_files and isinstance(objekt, (str, bytes)) and
                        os.path.isfile(objekt)):
                    if hash_method is None:
                        hash_method = config.get('execution', 'hash_method')

                    if hash_method.lower() == 'timestamp':
                        hash = hash_timestamp(objekt)
                    elif hash_method.lower() == 'content':
                        hash = hash_infile(objekt)
                    else:
                        raise Exception(
                            "Unknown hash method: %s" % hash_method)
                    if dictwithhash:
                        out = (objekt, hash)
                    else:
                        out = hash
                elif isinstance(objekt, float):
                    out = FLOAT_FORMAT(objekt)
                else:
                    out = objekt
        return out

    @property
    def __all__(self):
        return self.copyable_trait_names()


class TraitedSpec(BaseTraitedSpec):
    """ Create a subclass with strict traits.

    This is used in 90% of the cases.
    """
    _ = traits.Disallow


class BaseInterfaceInputSpec(TraitedSpec):
    pass


class DynamicTraitedSpec(BaseTraitedSpec):
    """ A subclass to handle dynamic traits

    This class is a workaround for add_traits and clone_traits not
    functioning well together.
    """

    def __deepcopy__(self, memo):
        """ bug in deepcopy for HasTraits results in weird cloning behavior for
        added traits
        """
        id_self = id(self)
        if id_self in memo:
            return memo[id_self]
        dup_dict = deepcopy(self.trait_get(), memo)
        # access all keys
        for key in self.copyable_trait_names():
            if key in self.__dict__.keys():
                _ = getattr(self, key)
        # clone once
        dup = self.clone_traits(memo=memo)
        for key in self.copyable_trait_names():
            try:
                _ = getattr(dup, key)
            except:
                pass
        # clone twice
        dup = self.clone_traits(memo=memo)
        dup.trait_set(**dup_dict)
        return dup


class CommandLineInputSpec(BaseInterfaceInputSpec):
    args = traits.Str(argstr='%s', desc='Additional parameters to the command')
    environ = traits.DictStrStr(
        desc='Environment variables', usedefault=True, nohash=True)


class StdOutCommandLineInputSpec(CommandLineInputSpec):
    out_file = traits.File(argstr="> %s", position=-1, genfile=True)


class MpiCommandLineInputSpec(CommandLineInputSpec):
    use_mpi = traits.Bool(
        False,
        desc="Whether or not to run the command with mpiexec",
        usedefault=True)
    n_procs = traits.Int(desc="Num processors to specify to mpiexec. Do not "
                         "specify if this is managed externally (e.g. through "
                         "SGE)")


def get_metadata(input_spec, metadata_key):
    """Query traits with a given metadata not None"""
    metadata = {metadata_key: lambda t: t is not None}
    info = [{'key': name, metadata_key: traitobj.copyfile}
        for name, traitobj in sorted(input_spec().traits(**metadata).items())
    ]
    return info


def get_filecopy_info(input_spec):
    """Provides information about file inputs to copy or link to cwd.
    Necessary for pipeline operation
    """
    metadata = dict(copyfile=lambda t: t is not None)
    for name, traitobj in sorted(input_spec().traits(**metadata).items()):
        info.append(dict(key=name, copy=traitobj.copyfile))
    return info


def check_inputs(inputs, raise_exception=False):
    """
    Check a traited spec of inputs: xor'ed, require'd and version'ed inputs.

    Parameters
    ----------

    inputs : InputSpec
        InputSpec object to be checked
    raise_exception : bool
        If ``True`` an exception will be issued, warnings otherwise.
    """

    return True


    def _check_mandatory_inputs(self):
        """ Raises an exception if a mandatory input is Undefined
        """
        for name, spec in list(self.inputs.traits(mandatory=True).items()):
            value = getattr(self.inputs, name)
            self._check_xor(spec, name, value)
            # Check xor
            values = [
                isdefined(getattr(self.inputs, field)) for field in spec.xor
            ]
            if not isdefined(value) and spec.xor is None:
                msg = ("%s requires a value for input '%s'. "
                       "For a list of required inputs, see %s.help()" %
                       (self.__class__.__name__, name,
                        self.__class__.__name__))
                raise ValueError(msg)
            if isdefined(value) and spec.requires:
                requires_list = [
                    not isdefined(getattr(self.inputs, field))
                    for field in spec.requires
                ]
        for name, spec in list(
                self.inputs.traits(mandatory=None, transient=None).items()):
            self._check_requires(spec, name, getattr(self.inputs, name))

    def _check_version_requirements(self, trait_object, raise_exception=True):
        """Raises an exception on version mismatch"""
        unavailable_traits = []
        # check minimum version
        check = dict(min_ver=lambda t: t is not None)
        names = trait_object.trait_names(**check)

        if names and self.version:
            version = LooseVersion(str(self.version))
            for name in names:
                min_ver = LooseVersion(
                    str(trait_object.traits()[name].min_ver))
                if min_ver > version:
                    unavailable_traits.append(name)
                    if not isdefined(getattr(trait_object, name)):
                        continue
                    if raise_exception:
                        raise Exception(
                            'Trait %s (%s) (version %s < required %s)' %
                            (name, self.__class__.__name__, version, min_ver))

        # check maximum version
        check = dict(max_ver=lambda t: t is not None)
        names = trait_object.trait_names(**check)
        if names and self.version:
            version = LooseVersion(str(self.version))
            for name in names:
                max_ver = LooseVersion(
                    str(trait_object.traits()[name].max_ver))
                if max_ver < version:
                    unavailable_traits.append(name)
                    if not isdefined(getattr(trait_object, name)):
                        continue
                    if raise_exception:
                        raise Exception(
                            'Trait %s (%s) (version %s > required %s)' %
                            (name, self.__class__.__name__, version, max_ver))
        return unavailable_traits

    def _list_outputs(self):
        """ List the expected outputs
        """
        if self.output_spec:
            raise NotImplementedError

        return {}

    def aggregate_outputs(self, runtime=None, needed_outputs=None):
        """ Collate expected outputs and check for existence
        """
        listed_outputs = self._list_outputs()

        # See whether there are outputs set
        predicted_outputs = set(list(listed_outputs.keys()))
        if not predicted_outputs:
            return outputs

        needed_outputs = set(needed_outputs or [])

        # Check whether some outputs are unavailable
        unavailable_outputs = []
        if outputs:
            unavailable_outputs = self._check_version_requirements(
                outputs, raise_exception=False)
        if predicted_outputs.intersection(unavailable_outputs):
            raise KeyError((
                'Output traits %s are not available in version '
                '%s of interface %s. Please inform developers.') % (
                ', '.join(['"%s"' % s for s in unavailable_outputs]), self.version,
                          self.__class__.__name__))

        for name in predicted_outputs.intersection(needed_outputs):
            value = listed_outputs[name]
            try:
                setattr(outputs, name, value)
            except TraitError as error:
                if getattr(error, 'info',
                           'default').startswith('an existing'):
                    raise FileNotFoundError(
                        "File/Directory '%s' not found for %s output "
                        "'%s'." % (value, self.__class__.__name__, name))
                raise error

        return outputs




    def load_inputs_from_json(self, json_file, overwrite=True):
        """A convenient way to load pre-set inputs from a JSON file."""

        with open(json_file) as fhandle:
            inputs_dict = json.load(fhandle)

        def_inputs = []
        if not overwrite:
            def_inputs = list(self.inputs.get_traitsfree().keys())

        new_inputs = [i for i in set(list(inputs_dict.keys())) - set(def_inputs)]
        for key in new_inputs:
            if hasattr(self.inputs, key):
                setattr(self.inputs, key, inputs_dict[key])

    def save_inputs_to_json(self, json_file):
        """A convenient way to save current inputs to a JSON file."""
        inputs = self.inputs.get_traitsfree()
        iflogger.debug('saving inputs %s', inputs)
        with open(json_file, 'w' if PY3 else 'wb') as fhandle:
            json.dump(inputs, fhandle, indent=4, ensure_ascii=False)

    def _format_arg(self, name, trait_spec, value):
        """A helper function for _parse_inputs

        Formats a trait containing argstr metadata
        """
        argstr = trait_spec.argstr
        iflogger.debug('%s_%s', name, value)
        if trait_spec.is_trait_type(traits.Bool) and "%" not in argstr:
            # Boolean options have no format string. Just append options if True.
            return argstr if value else None
        # traits.Either turns into traits.TraitCompound and does not have any
        # inner_traits
        elif trait_spec.is_trait_type(traits.List) \
            or (trait_spec.is_trait_type(traits.TraitCompound) and
                isinstance(value, list)):
            # This is a bit simple-minded at present, and should be
            # construed as the default. If more sophisticated behavior
            # is needed, it can be accomplished with metadata (e.g.
            # format string for list member str'ification, specifying
            # the separator, etc.)

            # Depending on whether we stick with traitlets, and whether or
            # not we beef up traitlets.List, we may want to put some
            # type-checking code here as well
            sep = trait_spec.sep if trait_spec.sep is not None else ' '

            if argstr.endswith('...'):
                # repeatable option
                # --id %d... will expand to
                # --id 1 --id 2 --id 3 etc.,.
                argstr = argstr.replace('...', '')
                return sep.join([argstr % elt for elt in value])
            else:
                return argstr % sep.join(str(elt) for elt in value)
        else:
            # Append options using format string.
            return argstr % value

    def _filename_from_source(self, name, chain=None):
        if chain is None:
            chain = []

        trait_spec = self.inputs.trait(name)
        retval = getattr(self.inputs, name)
        source_ext = None
        if not isdefined(retval) or "%s" in retval:
            if not trait_spec.name_source:
                return retval

            # Do not generate filename when excluded by other inputs
            if any(isdefined(getattr(self.inputs, field))
                   for field in trait_spec.xor or ()):
                return retval

            # Do not generate filename when required fields are missing
            if not all(isdefined(getattr(self.inputs, field))
                       for field in trait_spec.requires or ()):
                return retval

            if isdefined(retval) and "%s" in retval:
                name_template = retval
            else:
                name_template = trait_spec.name_template
            if not name_template:
                name_template = "%s_generated"

            ns = trait_spec.name_source
            while isinstance(ns, (list, tuple)):
                if len(ns) > 1:
                    iflogger.warning(
                        'Only one name_source per trait is allowed')
                ns = ns[0]

            if not isinstance(ns, (str, bytes)):
                raise ValueError(
                    'name_source of \'{}\' trait should be an input trait '
                    'name, but a type {} object was found'.format(
                        name, type(ns)))

            if isdefined(getattr(self.inputs, ns)):
                name_source = ns
                source = getattr(self.inputs, name_source)
                while isinstance(source, list):
                    source = source[0]

                # special treatment for files
                try:
                    _, base, source_ext = split_filename(source)
                except (AttributeError, TypeError):
                    base = source
            else:
                if name in chain:
                    raise NipypeInterfaceError(
                        'Mutually pointing name_sources')

                chain.append(name)
                base = self._filename_from_source(ns, chain)
                if isdefined(base):
                    _, _, source_ext = split_filename(base)
                else:
                    # Do not generate filename when required fields are missing
                    return retval

            chain = None
            retval = name_template % base
            _, _, ext = split_filename(retval)
            if trait_spec.keep_extension and (ext or source_ext):
                if (ext is None or not ext) and source_ext:
                    retval = retval + source_ext
            else:
                retval = self._overload_extension(retval, name)
        return retval

    def _gen_filename(self, name):
        raise NotImplementedError

    def _overload_extension(self, value, name=None):
        return value


    def _parse_inputs(self, skip=None):
        """Parse all inputs using the ``argstr`` format string in the Trait.

        Any inputs that are assigned (not the default_value) are formatted
        to be added to the command line.

        Returns
        -------
        all_args : list
            A list of all inputs formatted for the command line.

        """
        all_args = []
        initial_args = {}
        final_args = {}
        metadata = dict(argstr=lambda t: t is not None)
        for name, spec in sorted(self.inputs.traits(**metadata).items()):
            if skip and name in skip:
                continue
            value = getattr(self.inputs, name)
            if spec.name_source:
                value = self._filename_from_source(name)
            elif spec.genfile:
                if not isdefined(value) or value is None:
                    value = self._gen_filename(name)

            if not isdefined(value):
                continue
            arg = self._format_arg(name, spec, value)
            if arg is None:
                continue
            pos = spec.position
            if pos is not None:
                if int(pos) >= 0:
                    initial_args[pos] = arg
                else:
                    final_args[pos] = arg
            else:
                all_args.append(arg)
        first_args = [el for _, el in sorted(initial_args.items())]
        last_args = [el for _, el in sorted(final_args.items())]
        return first_args + all_args + last_args