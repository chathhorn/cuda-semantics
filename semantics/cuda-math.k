load dynamic-c-semantics

module CUDA-MATH is
      including DYNAMIC-C-SEMANTICS

	rule [sqrtf]:
		<k> prepareBuiltin((Identifier("sqrtf")), tv(F:Float, t(_, float)))
			=> tv(sqrtFloat(F:Float), t(.Set, float))
		...</k>
		[structural]

	rule [logf]:
		<k> prepareBuiltin((Identifier("logf")), tv(F:Float, t(_, float)))
			=> tv(logFloat(F:Float), t(.Set, float))
		...</k>
		[structural]

	rule [expf]:
		<k> prepareBuiltin((Identifier("expf")), tv(F:Float, t(_, float)))
			=> tv(expFloat(F:Float), t(.Set, float))
		...</k>
		[structural]

	rule [atanf]:
		<k> prepareBuiltin((Identifier("atanf")), tv(F:Float, t(_, float)))
			=> tv(atanFloat(F:Float), t(.Set, float))
		...</k>
		[structural]

	rule [asinf]:
		<k> prepareBuiltin((Identifier("asinf")), tv(F:Float, t(_, float)))
			=> tv(asinFloat(F:Float), t(.Set, float))
		...</k>
		[structural]

	rule [atan2f]:
		<k> prepareBuiltin((Identifier("atan2f")), (tv(F:Float, t(_, float)),, tv(F':Float, t(_, float))))
			=> tv(atan2Float(F:Float, F':Float), t(.Set, float))
		...</k>
		[structural]

	rule [tanf]:
		<k> prepareBuiltin((Identifier("tanf")), tv(F:Float, t(_, float)))
			=> tv(tanFloat(F:Float), t(.Set, float))
		...</k>
            [structural]

	rule [floorf]:
		<k> prepareBuiltin((Identifier("floorf")), tv(F:Float, t(_, float)))
			=> tv(floorFloat(F:Float), t(.Set, float))
		...</k>
		[structural]

	rule [cosf]:
		<k> prepareBuiltin((Identifier("cosf")), tv(F:Float, t(_, float)))
			=> tv(cosFloat(F:Float), t(.Set, float))
		...</k>
		[structural]

	rule [fmodf]:
		<k> prepareBuiltin((Identifier("fmodf")), (tv(F:Float, t(_, float)),, tv(F':Float, t(_, float))))
			=> tv(F:Float %Float F':Float, t(.Set, float))
		...</k>
		[structural]

	rule [sinf]:
		<k> prepareBuiltin((Identifier("sinf")), tv(F:Float, t(_, float)))
			=> tv(sinFloat(F:Float), t(.Set, float))
		...</k>
		[structural]

      /*@ Single-precision intrinsics. */

	rule [__cosf]:
		<k> prepareBuiltin((Identifier("__cosf")), tv(F:Float, t(_, float)))
			=> tv(cosFloat(F:Float), t(.Set, float))
		...</k>
		[structural]

	rule [__expf]:
		<k> prepareBuiltin((Identifier("__expf")), tv(F:Float, t(_, float)))
			=> tv(expFloat(F:Float), t(.Set, float))
		...</k>
		[structural]

	rule [__logf]:
		<k> prepareBuiltin((Identifier("__logf")), tv(F:Float, t(_, float)))
			=> tv(logFloat(F:Float), t(.Set, float))
		...</k>
		[structural]

	rule [__sinf]:
		<k> prepareBuiltin((Identifier("__sinf")), tv(F:Float, t(_, float)))
			=> tv(sinFloat(F:Float), t(.Set, float))
		...</k>
		[structural]

	rule [__tanf]:
		<k> prepareBuiltin((Identifier("__tanf")), tv(F:Float, t(_, float)))
			=> tv(tanFloat(F:Float), t(.Set, float))
		...</k>
            [structural]

end module
