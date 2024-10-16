import { Request } from 'express';
import {
  Controller,
  Get,
  Query,
  Req,
  UnauthorizedException,
  UseGuards,
  ValidationPipe,
} from '@nestjs/common';

import { UsersService } from '../users.service';
import { JwtGuard } from 'src/auth/guards/jwt.guard';

import { GetAllUsersQuery } from '../dto/users.dto';
import type { Payload } from 'src/types/payload.type';

@Controller('/users/admins')
export class AdminsController {
  constructor(private readonly usersService: UsersService) {}

  @UseGuards(JwtGuard)
  @Get()
  async getAllAdmins(
    @Req() request: Request,
    @Query(new ValidationPipe({ transform: true }))
    query: GetAllUsersQuery,
  ) {
    const user: Payload = request['user'];

    if (user.role === 'patient') {
      throw new UnauthorizedException();
    }

    return this.usersService.findAll(user.isSuperAdmin, 'admin', {
      ...query,
    });
  }
}
